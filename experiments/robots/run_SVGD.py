import sys, os, logging, torch, math
from datetime import datetime
from torch.utils.data import DataLoader
from pyro.distributions import Normal
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from utils.plot_functions import *
from plants import RobotsSystem, RobotsDataset
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from arg_parser import argument_parser, print_args
from inference_algs.distributions import GibbsPosterior
from controllers import PerfBoostController, AffineController, NNController, SVGDCont

"""

"""
def train_svgd(args, logger, save_folder):
    TRAIN_METHOD = 'SVGD'

    msg = print_args(args)

    logger.info('---------- ' + TRAIN_METHOD + ' ----------\n\n')
    logger.info(msg)
    torch.manual_seed(args.random_seed)

    save_path = save_folder
    while not save_path.endswith('saved_results'):
        save_path = str(Path(save_path).parent.absolute())

    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
    # divide to train and test
    train_data_full, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_data_full, test_data = train_data_full.to(device), test_data.to(device)
    # validation data
    if args.early_stopping or args.return_best:
        valid_inds = torch.randperm(train_data_full.shape[0])[:int(args.validation_frac*train_data_full.shape[0])]
        train_inds = [ind for ind in range(train_data_full.shape[0]) if ind not in valid_inds]
        valid_data = train_data_full[valid_inds, :, :]
        train_data = train_data_full[train_inds, :, :]
    else:
        valid_data = None
        train_data = train_data_full
    # data for plots
    t_ext = args.horizon * 4
    plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
    plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
    plot_data = plot_data.to(device)
    # batch the data
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # ------------ 2. Plant ------------
    plant_input_init = None     # all zero
    plant_state_init = None     # same as xbar
    sys = RobotsSystem(
        xbar=dataset.xbar, x_init=plant_state_init,
        u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const
    ).to(device)

    # ------------ 3. Controller ------------
    if args.cont_type=='PerfBoost':
        ctl_generic = PerfBoostController(
            noiseless_forward=sys.noiseless_forward,
            input_init=sys.x_init, output_init=sys.u_init,
            dim_internal=args.dim_internal, dim_nl=args.dim_nl,
            initialization_std=args.cont_init_std,
            output_amplification=20, train_method=TRAIN_METHOD
        ).to(device)
    elif args.cont_type=='Affine':
        ctl_generic = AffineController(
            weight=torch.zeros(sys.in_dim, sys.state_dim, device=device, dtype=torch.float32),
            bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32),
            train_method=TRAIN_METHOD
        )
    elif args.cont_type=='NN':
        ctl_generic = NNController(
            in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=args.layer_sizes,
            train_method=TRAIN_METHOD
        )
    else:
        raise KeyError('[Err] args.cont_type must be PerfBoost, NN, or Affine.')

    num_params = ctl_generic.num_params
    logger.info('[INFO] Controller is of type ' + args.cont_type + ' and has %i parameters.' % num_params)

    # ------------ 4. Loss ------------
    Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
    x0 = dataset.x0.reshape(1, -1).to(device)
    sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
    sat_bound += 0 if args.alpha_col is None else args.alpha_col
    sat_bound += 0 if args.alpha_obst is None else args.alpha_obst
    sat_bound = sat_bound/20
    logger.info('Loss saturates at: '+str(sat_bound))
    bounded_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
        loss_bound=args.loss_bound, sat_bound=sat_bound.to(device),
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )
    original_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
        loss_bound=None, sat_bound=None,
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )

    # ------------ 5. Prior ------------
    if args.cont_type in ['Affine', 'NN']:
        training_param_names = ['weight', 'bias']
        prior_dict = {
            'type':'Gaussian', 'type_w':'Gaussian',
            'type_b':'Gaussian_biased',
            'weight_loc':0, 'weight_scale':1,
            'bias_loc':0, 'bias_scale':5,
        }
    else:
        if args.data_dep_prior:
            if args.dim_nl==8 and args.dim_internal==8:
                if args.num_rollouts_prior==5:
                    filename_load = os.path.join(save_path, 'empirical', 'pretrained', 'trained_controller.pt')
                    res_dict_loaded = torch.load(filename_load)
        if args.nominal_prior:
            res_dict_loaded = []
            if args.dim_nl==8 and args.dim_internal==8:
                for _, dirs, _ in os.walk(os.path.join(save_path, 'nominal')):
                    for dir in dirs:
                        filename_load = os.path.join(save_path, 'nominal', dir, 'trained_controller.pt')
                        res_dict_loaded.append(torch.load(filename_load))
            logger.info('[INFO] Loaded '+str(len(res_dict_loaded))+' nominal controllers.')
        prior_dict = {'type':'Gaussian'}
        training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        for name in training_param_names:
            if args.data_dep_prior:
                prior_dict[name+'_loc'] = res_dict_loaded[name]
                prior_dict[name+'_scale'] = args.prior_std
            elif args.nominal_prior:
                logger.info('[INFO] Prior distribution is the distribution over nominal controllers, with std scaled by %.4f.' % args.nominal_prior_std_scale)
                vals = torch.stack([res[name] for res in res_dict_loaded], dim=0)
                # val and std computed elementwise. same shape as the training param
                prior_dict[name+'_loc'] = vals.mean(dim=0)  
                prior_dict[name+'_scale'] = vals.std(dim=0, correction=1) * args.nominal_prior_std_scale
            else:
                prior_dict[name+'_loc'] = 0
                prior_dict[name+'_scale'] = args.prior_std

    # ------------ 6. Posterior ------------
    gibbs_lambda_star = (8*args.num_rollouts*math.log(1/args.delta))**0.5   # lambda for Gibbs
    gibbs_lambda = gibbs_lambda_star
    logger.info('gibbs_lambda: %.2f' % gibbs_lambda + ' (use lambda_*)' if gibbs_lambda == gibbs_lambda_star else '')
    # define target distribution
    gibbs_posterior = GibbsPosterior(
        loss_fn=bounded_loss_fn, lambda_=gibbs_lambda, prior_dict=prior_dict,
        # attributes of the CL system
        controller=ctl_generic, sys=sys,
        # misc
        logger=logger,
    )

    # ****** INIT SVGD ******
    # initialize trainable params
    dim = (args.num_particles, ctl_generic.num_params)

    if args.init_from_prior:
        logger.info('[INFO] Initializing particles from prior.')
        INIT_PARTICLE_STD_SCALE = 1
        PRIOR_STDP_SCALE = 50
        initial_particles = Normal(
            gibbs_posterior.prior.mean().reshape(-1), 
            gibbs_posterior.prior.stddev().reshape(-1)/PRIOR_STDP_SCALE/INIT_PARTICLE_STD_SCALE
        ).sample([args.num_particles]).to(device)
    else:
        initial_particles = Normal(0, args.cont_init_std).sample(dim).to(device)

    svgd_cont = SVGDCont(
        gibbs_posterior=gibbs_posterior,
        num_particles=args.num_particles, logger=logger,
        optimizer='Adam', lr=args.lr, lr_decay=None, #TODO: add decay
        initial_particles=initial_particles, kernel='RBF', bandwidth=None,
    )
    logger.info('\n[INFO] SVGD: delta: %.2f' % args.delta + ' -- num particles: %2.f' % args.num_particles)


    # ****** TRAIN SVGD ******

    logger.info('------------ Begin training ------------')
    svgd_cont.fit(
        train_dataloader=train_dataloader, 
        early_stopping=args.early_stopping, tol_percentage=args.tol_percentage, n_logs_no_change=args.n_logs_no_change,
        return_best=args.return_best, log_period=args.log_epoch, epochs=args.epochs,
        valid_data=valid_data, loss_fn=original_loss_fn, save_folder=save_folder
    )
    logger.info('Training completed.')

    # ------ Save trained model ------
    particles = svgd_cont.particles.detach().clone()
    res_dict = args.__dict__
    res_dict['particles'] = particles
    res_dict['Q'] = Q
    # save file name
    filename_save = os.path.join(save_folder, 'trained_model')
    torch.save(res_dict, filename_save)
    logger.info('model saved.')

    # ------------ 5. Test Dataset ------------
    # eval on train data
    bounded_train_loss = svgd_cont.eval_rollouts(train_data_full, count_collisions=False)
    original_train_loss, train_num_cols = svgd_cont.eval_rollouts(train_data_full, loss_fn=original_loss_fn, count_collisions=args.col_av)
    msg = 'Final results on the entire train data: Bounded train loss = {:.4f}, original train loss = {:.4f}'.format(
        bounded_train_loss, original_train_loss
    )
    res_dict['train_loss'] = original_train_loss
    if args.col_av:
        msg += ', train num collisions = {:3.0f}'.format(train_num_cols.item())
        res_dict['train_num_cols'] = train_num_cols.item()
    # eval on test data
    bounded_test_loss = svgd_cont.eval_rollouts(test_data, count_collisions=False)
    original_test_loss, test_num_cols = svgd_cont.eval_rollouts(test_data, loss_fn=original_loss_fn, count_collisions=args.col_av)
    msg += 'True bounded test loss = {:.4f}, '.format(bounded_test_loss)
    msg += 'true original test loss = {:.2f} '.format(original_test_loss)
    msg += '(approximated using {:3.0f} test rollouts).'.format(test_data.shape[0])
    res_dict['test_loss'] = original_test_loss
    if args.col_av:
        msg += ', test num collisions = {:3.0f}'.format(test_num_cols.item())
        res_dict['test_num_cols'] = test_num_cols.item()
    logger.info(msg)

    return res_dict, filename_save


if __name__=='__main__':
    # ----- parse and set experiment arguments -----
    args = argument_parser()
    # ----- SET UP LOGGER -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
    save_folder = os.path.join(save_path, 'SVGD', args.cont_type+'_'+now)
    os.makedirs(save_folder)
    logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger('ren_controller_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)
    # ----- run experiment -----
    train_svgd(args, logger, save_folder)