import argparse, math

# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Training ren for learning contractive motion through imitation.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')
    parser.add_argument('--col-av', type=str2bool, default=True, help='Avoid collisions. Default is True.')
    parser.add_argument('--obst-av', type=str2bool, default=True, help='Avoid obstacles. Default is True.')

    # dataset
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the computation. Default is 100.')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents. Default is 2.')
    parser.add_argument('--num-rollouts', type=int, default=30, help='Number of rollouts in the training data. Default is 30.')
    parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0 , help='Spring constant. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=str2bool, default=False, help='Linearize plant or not. Default is False.')

    # controller
    parser.add_argument('--cont-type', type=str, default='Affine', help='Controller type. Can be Affine, NN, or PerfBoost. Default is Affine.')
    parser.add_argument('--cont-init-std', type=float, default=0.1 , help='Initialization std for controller params. Default is 0.1.')
    # PerfBoost nontroller
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')
    # NN controller
    parser.add_argument('--layer-sizes', nargs='+', default=None, help='size of NN controller hidden layers. Default is no hidden layers. use like --layer-sizes 4 4 for 2 hidden layers each with 4 neurons.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400 , help='Weight of the loss due to control input "u". Default is 0.1/400.') #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=100 , help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3 , help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--min-dist', type=float, default=1.0 , help='TODO. Default is 1.0 if "col-av" is True, else None.') #TODO: add help

    # optimizer
    parser.add_argument('--batch-size', type=int, default=5, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    parser.add_argument('--epochs', type=int, default=-1, help='Total number of epochs for training. Default is 5000 if collision avoidance, else 100.')
    parser.add_argument('--lr', type=float, default=-1, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay for Adam optimizer. Default is 0.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best', type=str2bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. Default is True.')

    # inference
    parser.add_argument('--flow-type', type=str, default='Planar', help='Flow type for normflow. Can be Planar, Radial, or NVP. Default is Planar.')
    parser.add_argument('--flow-activation', type=str, default='tanh', help='Activation function of each flow for normflow. Can be tanh or leaky_relu. Default is tanh.')
    parser.add_argument('--num-flows', type=int, default=16, help='Number of transforms in for normflow. Default is 16. Set to 0 for no transforms')
    parser.add_argument('--base-is-prior', type=str2bool, default=False, help='Base distribution for normflow is the same as the prior. Default is False.')
    parser.add_argument('--base-center-emp', type=str2bool, default=False, help='Base distribution for normflow is centered at the controller learned empirically. Default is False.')
    parser.add_argument('--learn-base', type=str2bool, default=True, help='Optimize base distribution of normflow. Default is True.')
    parser.add_argument('--prior-std', type=float, default=7, help='Gaussian prior std. Default is 7.')
    parser.add_argument('--annealing', type=str2bool, default=True, help='Annealing loss for normflow. Default is False.')
    parser.add_argument('--anneal-iter', type=int, default=None, help='Annealing iteration for normflow. Default is half epochs.')

    # Gibbs
    parser.add_argument('--delta', type=float, default=0.1 , help='Delta for Gibbs distribution. PAC bounds hold with prob >= 1- delta. Default is 0.1.')
    parser.add_argument('--gibbs-lambda', type=float, default=None , help='Lambda is the tempretaure of the Gibbs distribution. Default is lambda_star (see the paper).')

    # TODO: add the following
    # parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    # parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    # parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    # # save/load args
    # parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')
    # parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')
    # parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')


    args = parser.parse_args()

    # set default values that depend on other args
    if args.batch_size==-1:
        args.batch_size = args.num_rollouts # use all train data

    if args.epochs==-1 or args.epochs is None:
        args.epochs = 1000 if args.col_av else 50

    if args.lr==-1 or args.lr is None:
        args.lr = 2e-3 if args.col_av else 5e-3

    if args.log_epoch==-1 or args.log_epoch is None:
        args.log_epoch = math.ceil(float(args.epochs)/10)

    if args.annealing and args.anneal_iter is None:
        args.anneal_iter = int(args.epochs/2)

    assert not (args.base_is_prior and args.base_center_emp)

    if args.gibbs_lambda is None:
        args.gibbs_lambda = (8*args.num_rollouts*math.log(1/args.delta))**0.5

    # assertions and warning
    if not args.col_av:
        args.alpha_col = None
        # args.min_dist = None
    if not args.obst_av:
        args.alpha_obst = None

    # if args.total_epochs < 10000:
    #     print(f'Minimum of 10000 epochs are required for proper training')

    if args.horizon > 100:
        print(f'Long horizons may be unnecessary and pose significant computation')

    if args.layer_sizes is None:
        args.layer_sizes = []
    else:
        args.layer_sizes = [int(i) for i in args.layer_sizes]

    return args


def print_args(args, method='empirical'):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller: cont_init_std: %.2f'% args.cont_init_std
    if args.cont_type=='PerfBoost':
        msg += ' -- dimension of the internal state: %i' % args.dim_internal
        msg += ' -- dim_nl: %i' % args.dim_nl

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u
    if args.col_av:
        msg += ' -- alpha_col: %.f' % args.alpha_col
    else:
        msg += ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr + 'weight decay: %.2e' % args.weight_decay
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    msg += '\n[INFO] Prior: prior std: %.2e' % args.prior_std

    msg += '\n[INFO] Gibbs: delta: %.2e' % args.delta + ' -- gibbs_lambda: %.2f' % args.gibbs_lambda

    # arguments for normflow:
    if method=='normflow':
        msg += '\n[INFO] Norm flows setup: num transformations: %i' % args.num_flows
        msg += ' -- flow type: ' + args.flow_type if args.num_flows>0 else ' -- flow type: None'
        msg += ' -- flow activation: ' + args.flow_activation
        msg += ' -- base dist: DiagGaussian -- base is prior: ' + str(args.base_is_prior)
        msg += ' -- base centered at emp: ' + str(args.base_center_emp) + ' -- learn base: ' + str(args.learn_base)
        msg += ' -- annealing: ' + str(args.annealing)
        msg += ' -- annealing iter: %i' % args.anneal_iter if args.annealing else ''

    return msg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'T', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'F', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')