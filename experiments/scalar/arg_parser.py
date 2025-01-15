import argparse, math


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Scalar experiment.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=33, help='Random seed. Default is 33.')
    # parser.add_argument('--col-av', type=bool, default=True, help='Avoid collisions. Default is True.')
    # parser.add_argument('--obst-av', type=bool, default=True, help='Avoid obstacles. Default is True.')

    # dataset
    parser.add_argument('--horizon', type=int, default=10, help='Time horizon for the computation. Default is 10.')
    # parser.add_argument('--n-agents', type=int, default=2, help='Number of agents. Default is 2.')
    parser.add_argument('--num-rollouts', type=int, default=512, help='Number of rollouts in the training data. Default is 512.')
    # parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.')

    # # plant
    # parser.add_argument('--spring-const', type=float, default=1.0 , help='Spring constant. Default is 1.0.')
    # parser.add_argument('--linearize-plant', type=bool, default=False, help='Linearize plant or not. Default is False.')

    # PerfBoost controller
    parser.add_argument('--cont-type', type=str, default='Affine', help='Controller type. Can be Affine or PerfBoost. Default is Affine.')
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')

    # loss
    parser.add_argument('--loss-bound', type=float, default=1.0, help='Bound the loss to this value. Default is 1.')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=8, help='Number of forward trajectories of the closed-loop system at each step. Default is 8.')
    parser.add_argument('--epochs', type=int, default=20000, help='Total number of epochs for training. Default is 20000.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 1e-3.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay. Default is 0.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.05 * epochs.')
    parser.add_argument('--return-best', type=bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')

    # Gibbs
    parser.add_argument('--delta', type=float, default=0.1 , help='Delta for Gibbs distribution. PAC bounds hold with prob >= 1- delta. Default is 0.1.')

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
    if args.batch_size == -1:
        args.batch_size = args.num_rollouts  # use all train data

    if args.log_epoch == -1 or args.log_epoch is None:
        args.log_epoch = math.ceil(float(args.epochs)/20)

    if args.horizon > 100:
        print(f'Long horizons may be unnecessary and pose significant computation')

    return args


def print_args(args):
    msg = '------------------ SCALAR EXP ------------------'
    msg += '\n[INFO] Dataset: horizon: %i' % args.horizon + ' -- num_rollouts: %i' % args.num_rollouts
    # msg += ' -- std_ini: %.2f' % args.std_init_plant

    if args.cont_type=='PerfBoost':
        msg += '\n[INFO] PerfBoost controller: dimension of the internal state: %i' % args.dim_internal
        msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f'% args.cont_init_std
    elif args.cont_type=='Affine':
        msg += '\n[INFO] Affine controller'
    elif args.cont_type=='NN':
        msg += '\n[INFO] NN controller'
    else:
        raise NotImplementedError('[Err] Only Affine, NN, and PerfBoost controllers are implemented.')

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr + ' -- weight_decay: %.4f' % args.weight_decay
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return_best (return best model for validation data among logged epochs): ' + str(args.return_best)

    return msg
