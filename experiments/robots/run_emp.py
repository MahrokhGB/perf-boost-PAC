import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger

def train_emp(args):
    TRAIN_METHOD = 'empirical'
    
    # ----- SET UP LOGGER -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    if args.nominal_exp:
        save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'nominal')
    else:
        save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'empirical')
    save_folder = os.path.join(save_path, args.cont_type+'_'+now)
    os.makedirs(save_folder)
    logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger('perf_boost_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)
    msg = print_args(args, TRAIN_METHOD)
    logger.info(msg)

    torch.manual_seed(args.random_seed)

    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
    if args.nominal_exp:
        # generate train and test
        _, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
        train_data_full = torch.zeros(1, args.horizon, test_data.shape[-1], device=device)
        train_data_full[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
        train_data_full, test_data = train_data_full.to(device), test_data.to(device)
        # validation data
        if args.early_stopping or args.return_best:
            logger.info('Using train data for early_stopping or return_best.')
            valid_data = train_data_full
        else:
            valid_data = None
        train_data = train_data_full
    else:
        # generate train and test
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
        ).to(device)
    elif args.cont_type=='NN':
        ctl_generic = NNController(
            in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=args.layer_sizes,
            train_method=TRAIN_METHOD
        ).to(device)
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
        n_agents=sys.n_agents,
    )
    original_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
        loss_bound=None, sat_bound=None,
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents,
    )

    # ------------ 6. Training ------------
    # plot closed-loop trajectories before training the controller
    logger.info('Plotting closed-loop trajectories before training the controller...')
    x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
    plot_trajectories(
        x_log[0, :, :], # remove extra dim due to batching
        xbar=dataset.xbar, n_agents=sys.n_agents,
        save_folder=save_folder, filename='CL_init.pdf',
        text="CL - before training", T=t_ext,
        obstacle_centers=original_loss_fn.obstacle_centers,
        obstacle_covs=original_loss_fn.obstacle_covs,
        plot_collisions=True, min_dist=args.min_dist
    )
    ctl_generic.fit(
        sys=sys, train_dataloader=train_dataloader, valid_data=valid_data, 
        lr=args.lr, loss_fn=original_loss_fn, epochs=args.epochs, log_epoch=args.log_epoch, 
        return_best=args.return_best, logger=logger, early_stopping=args.early_stopping, 
        n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage
    )
    
    # ------ 7. Save and evaluate the trained model ------
    # save
    res_dict = ctl_generic.c_ren.state_dict()
    print('res_dict', res_dict['X'][0][0:5])
    res_dict['Q'] = Q
    filename = os.path.join(save_folder, 'trained_controller'+'.pt')
    torch.save(res_dict, filename)
    logger.info('[INFO] saved trained model.')

    # evaluate on the train data
    logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data_full.shape[0])
    with torch.no_grad():
        x_log, _, u_log = sys.rollout(
            controller=ctl_generic, data=train_data_full
        )   # use the entire train data, not a batch
        # evaluate losses
        loss = original_loss_fn.forward(x_log, u_log)
        msg = 'Loss: %.4f' % (loss)
    # count collisions
    if args.col_av:
        num_col = original_loss_fn.count_collisions(x_log)
        msg += ' -- Number of collisions = %i' % num_col
    logger.info(msg)

    # evaluate on the test data
    logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
    with torch.no_grad():
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl_generic, data=test_data
        )
        # loss
        test_loss = original_loss_fn.forward(x_log, u_log).item()
        msg = "Loss: %.4f" % (test_loss)
    # count collisions
    if args.col_av:
        num_col = original_loss_fn.count_collisions(x_log)
        msg += ' -- Number of collisions = %i' % num_col
    logger.info(msg)

    # plot closed-loop trajectories using the trained controller
    logger.info('Plotting closed-loop trajectories using the trained controller...')
    x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
    plot_trajectories(
        x_log[0, :, :], # remove extra dim due to batching
        xbar=dataset.xbar, n_agents=sys.n_agents,
        save_folder=save_folder, filename='CL_trained.pdf',
        text="CL - trained controller", T=t_ext,
        obstacle_centers=original_loss_fn.obstacle_centers,
        obstacle_covs=original_loss_fn.obstacle_covs,
        plot_collisions=True, min_dist=args.min_dist
    )

    return res_dict, filename



if __name__=='__main__':
    # ----- parse and set experiment arguments -----
    args = argument_parser()
    train_emp(args)