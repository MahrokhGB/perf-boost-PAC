import sys, os, logging, torch, math
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import normflows as nf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import fit_norm_flow
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist.mynf import NormalizingFlow
from inference_algs.normflow_assist import GibbsWrapperNF

def train_normflow(args, logger, save_folder):
    TRAIN_METHOD = 'normflow'

    msg = print_args(args, TRAIN_METHOD)

    logger.info('---------- ' + TRAIN_METHOD + ' ----------\n\n')
    logger.info(msg)
    torch.manual_seed(args.random_seed)

    save_path = save_folder
    while not save_path.endswith('saved_results'):
        save_path = str(Path(save_path).parent.absolute())

    BASE_STD_SCALE = 1/args.nominal_prior_std_scale/100 # TODO

    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)
    # remove data used for training the prior
    if args.data_dep_prior:
        num_train_rollouts = args.num_rollouts - args.num_rollouts_prior
        train_data = train_data[args.num_rollouts_prior:, :]    # remove samples used for training the prior from the posterior train data
    else:
        num_train_rollouts = args.num_rollouts
    # data for plots
    t_ext = args.horizon * 4
    plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
    plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
    plot_data = plot_data.to(device)
    # batch the data
    # NOTE: for normflow, must use all the data at each iter b.c. otherwise, the target distribution changes.
    train_dataloader = DataLoader(train_data, batch_size=min(num_train_rollouts, 256), shuffle=False)

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
    logger.info('Loss saturates at: '+str(sat_bound)+'\n')
    bounded_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
        loss_bound=args.loss_bound, sat_bound=sat_bound.to(device),
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )
    # original_loss_fn = RobotsLossMultiBatch(
    #     Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    #     loss_bound=None, sat_bound=None,
    #     alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    #     min_dist=args.min_dist if args.col_av else None,
    #     n_agents=sys.n_agents if args.col_av else None,
    # )

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
                logger.info(
                    '[INFO] Prior distribution is the distribution over nominal controllers, with std scaled by %.4f.' % args.nominal_prior_std_scale
                )
                vals = torch.stack([res[name] for res in res_dict_loaded], dim=0)
                # val and std computed elementwise. same shape as the training param
                prior_dict[name+'_loc'] = vals.mean(dim=0)  
                prior_dict[name+'_scale'] = vals.std(dim=0, correction=1) * args.nominal_prior_std_scale
            else:
                prior_dict[name+'_loc'] = 0
                prior_dict[name+'_scale'] = args.prior_std

    # ------------ 6. Posterior ------------
    # use max lambda s.t. eps/lambda <= thresh
    # thresh_eps_lambda = 0.2
    # num_prior_samples = 10**6
    # lambda_max_eps = 1000
    # lambda_max_eps = get_max_lambda(thresh=thresh_eps_lambda, delta=args.delta, n_p=num_prior_samples, init_condition=20, loss_bound=args.loss_bound)    #TODO
    # logger.info('lambda_max_eps = '+str(lambda_max_eps))
    # define target distribution
    gibbs_posteior = GibbsPosterior(
        loss_fn=bounded_loss_fn,
        lambda_=args.gibbs_lambda, # lambda_max_eps,  # # TODO
        prior_dict=prior_dict,
        # attributes of the CL system
        controller=ctl_generic, sys=sys,
        # misc
        logger=logger,
    )

    # Wrap Gibbs distribution to be used in normflows
    target = GibbsWrapperNF(
        target_dist=gibbs_posteior, train_dataloader=train_dataloader,
        prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
    )

    # ------------ 7. save setup ------------
    setup_dict = vars(args)
    setup_dict['sat_bound'] = sat_bound
    setup_dict['loss_bound'] = args.loss_bound
    setup_dict['prior_dict'] = prior_dict
    setup_dict['gibbs_lambda'] = gibbs_posteior.lambda_
    setup_dict['base_std_scale'] = BASE_STD_SCALE
    torch.save(setup_dict, os.path.join(save_folder, 'setup'))

    # ------------ 8. NormFlows ------------
    num_samples_nf_train = 100
    num_samples_nf_eval = num_samples_nf_train 

    flows = []
    for _ in range(args.num_flows):
        if args.flow_type == 'Radial':
            flows += [nf.flows.Radial((num_params,), act=args.flow_activation)]
        elif args.flow_type == 'Planar': # f(z) = z + u * h(w * z + b)
            '''
            Default values:
                - u: uniform(-sqrt(2), sqrt(2))
                - w: uniform(-sqrt(2/num_prams), sqrt(2/num_prams))
                - b: 0
                - h: args.flow_activation (tanh or leaky_relu)
            '''
            flows += [nf.flows.Planar(
                (num_params,), u=args.planar_flow_scale*(2*torch.rand(num_params)-1), act=args.flow_activation
            )]
        elif args.flow_type == 'NVP':
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([math.ceil(num_params/2), 64, 64, num_params], init_zeros=True, act=args.flow_activation)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))
        else:
            raise NotImplementedError

    # base distribution
    q0 = nf.distributions.DiagGaussian(num_params, trainable=args.learn_base)
    # base distribution same as the prior
    if args.base_is_prior:
        logger.info('[INFO] Base distribution is similar to the prior, with std scaled by %.4f.' % BASE_STD_SCALE)
        state_dict = q0.state_dict()
        state_dict['loc'] = gibbs_posteior.prior.mean().reshape(1, -1)
        state_dict['log_scale'] = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1)*BASE_STD_SCALE) 
        q0.load_state_dict(state_dict)
    # base distribution centered at the empirical or nominal controller
    elif args.base_center_emp:
        # get filename to load
        if args.dim_nl==1 and args.dim_internal==1:
            filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_58_13', 'trained_controller.pt')
        elif args.dim_nl==2 and args.dim_internal==4:
            filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_57_38', 'trained_controller.pt')
        elif args.dim_nl==8 and args.dim_internal==8:
            # # empirical controller avoids collisions
            # filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_01_14_16_26_11', 'trained_controller.pt')
            # empirical controller does not avoid collisions
            filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_01_19_11_31_25', 'trained_controller.pt')
        # load the controller
        res_dict_loaded = torch.load(filename_load)
        # set the mean of the base distribution to the controller
        mean = np.array([])
        for name in training_param_names:
            mean = np.append(mean, res_dict_loaded[name].cpu().detach().numpy().flatten())
        state_dict = q0.state_dict()
        state_dict['loc'] = torch.Tensor(mean.reshape(1, -1))
        # set the scale of the base distribution
        # state_dict['log_scale'] = state_dict['log_scale'] - 100 # TODO
        state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*BASE_STD_SCALE)
        q0.load_state_dict(state_dict)
    else:
        state_dict = q0.state_dict()
        state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*BASE_STD_SCALE)
        q0.load_state_dict(state_dict)

    # set up normflow
    nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
    nfm.to(device)  # Move model on GPU if available

    # ------------ 10. Train NormFlows ------------
    optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    res_dict, filename_save = fit_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, logger=logger, loss_fn=bounded_loss_fn,
        save_folder=save_folder, train_data_full=train_data, test_data=test_data, plot_data=plot_data,
        return_best=args.return_best, validation_frac=args.validation_frac,
        early_stopping=args.early_stopping, n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage,
        optimizer=optimizer, epochs=args.epochs, log_epoch=args.log_epoch, annealing=args.annealing,
        anneal_iter=args.anneal_iter, num_samples_nf_train=num_samples_nf_train, num_samples_nf_eval=num_samples_nf_eval,
    )

    return res_dict, filename_save



if __name__=='__main__':
    # ----- parse and set experiment arguments -----
    args = argument_parser()
    # ----- SET UP LOGGER -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
    save_folder = os.path.join(save_path, 'normflow', args.cont_type+'_'+now)
    os.makedirs(save_folder)
    logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger('ren_controller_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)
    # ----- run experiment -----
    train_normflow(args, logger, save_folder)
