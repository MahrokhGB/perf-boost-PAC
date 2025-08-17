import sys, os, logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from utils.assistive_functions import WrapLogger
from arg_parser import argument_parser
from experiments.robots.run_normflow import train_normflow

# norm flow
# use lambda star and tune nominal prior std for best performance, then grid search over N_p for tightest bound 
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method normflow --num-rollouts 2048 --batch-size 256 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --base-is-prior True --flow-activation tanh --delta 0.1 --lr 5e-4 --random-seed 0
res_normflow = [
    {
        'num_rollouts':8,
        'nominal_prior_std_scale': 473.1147454034813,
        'Bounded train loss': 0.0838, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0909, 
        'original test loss': None, 
        'test num collisions': 58
    },
    {
        'num_rollouts':16,
        'nominal_prior_std_scale': 420.88108056283505,
        'Bounded train loss': 0.0801, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0854, 
        'original test loss': None, 
        'test num collisions': 4
    },
    {
        'num_rollouts':32,
        'nominal_prior_std_scale': 58.93082020462471,
        'Bounded train loss': 0.0823, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0841, 
        'original test loss': None, 
        'test num collisions': 11
    },
    {
        'num_rollouts':64,
        'nominal_prior_std_scale': 55.299916543094554,
        'Bounded train loss': 0.0840, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0856, 
        'original test loss': None, 
        'test num collisions': 40
    },
    {
        'num_rollouts':128,
        'nominal_prior_std_scale': 80.12092549898787,
        'Bounded train loss': 0.0823, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0822, 
        'original test loss': None, 
        'test num collisions': 0
    },
    {
        'num_rollouts':256,
        'nominal_prior_std_scale': 170.39857841504897,
        'Bounded train loss': 0.0830, 
        'original train loss': None,
        'train num collisions': 2,
        'bounded test loss': 0.0835, 
        'original test loss': None, 
        'test num collisions': 10
    },
    {
        'num_rollouts':512,
        'nominal_prior_std_scale': 99.42019860940434,
        'Bounded train loss': 0.0825, 
        'original train loss': None,
        'train num collisions': 1,
        'bounded test loss': 0.0827, 
        'original test loss': None, 
        'test num collisions': 3  
    },
    {
        'num_rollouts':1024,
        'nominal_prior_std_scale': 5.657081525352462,
        'Bounded train loss': 0.0864, 
        'original train loss': None,
        'train num collisions': 41,
        'bounded test loss': 0.0861, 
        'original test loss': None, 
        'test num collisions': 35
    },
    {
        'num_rollouts':2048,
        'nominal_prior_std_scale': 284.5670370215786,
        'Bounded train loss': None, 
        'original train loss': None,
        'train num collisions': None,
        'bounded test loss': 0.0826, 
        'original test loss': None, 
        'test num collisions': 3
    },
]


# ----- default experiment arguments -----
args = argument_parser()
args.random_seed = 0
args.num_rollouts = None # will be set later
args.batch_size = None # will be set later
args.nn_type = 'REN'
args.nominal_prior = True
args.base_is_prior = True
args.flow_activation = 'tanh'
args.nominal_prior_std_scale = None # will be set later

save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'ub_ub', 'saved_results')
setup_name ='internal' + str(args.dim_internal)
if args.nn_type == 'REN':
    setup_name += '_nl' + str(args.dim_nl)
elif args.nn_type == 'SSM':
    setup_name += '_middle' + str(args.dim_middle) + '_scaffolding' + str(args.dim_scaffolding)

for res in res_normflow:
    print(f"Num rollouts: {res['num_rollouts']}, Bounded test loss: {res['bounded test loss']}, Test num collisions: {res['test num collisions']}")
    args.num_rollouts = res['num_rollouts']
    args.batch_size = min(res['num_rollouts'], 256)
    args.nominal_prior_std_scale = res['nominal_prior_std_scale']
     # ----- SET UP LOGGER -----
    save_folder = os.path.join(save_path, 'normflow', args.nn_type, setup_name, args.cont_type+'_'+str(res['num_rollouts']))
    os.makedirs(save_folder)
    logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger('ren_controller_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)
    # train 
    res_dict, filename_save, nfm = train_normflow(args, logger, save_folder)

