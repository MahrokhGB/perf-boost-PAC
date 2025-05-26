import optuna, sys, os, logging
from datetime import datetime

from arg_parser import argument_parser
from run_emp import train_emp
from utils.assistive_functions import WrapLogger

def objective(trial):
    # ----- parse and set experiment arguments -----
    args = argument_parser()

    # Hyperparameters to tune
    # hidden_size = trial.suggest_int('hidden_size', 128, 512)
    args.cont_init_std = trial.suggest_float('cont_init_std', 1e-3, 1, log=True)
    args.lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    
    res_dict, _ = train_emp(args, logger, save_folder)
    print(res_dict['train_loss'])
    return res_dict['train_loss']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

# ----- SET UP LOGGER -----
method = 'empirical'
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'hyper_param_tuning')
save_folder = os.path.join(save_path, method+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

optuna.logging.disable_default_handler()  # Disable the default handler.
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
# ----- START OPTUNA STUDY -----
logger.info("Starting Hyperparameter Optimization with Optuna")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
logger.info("Best Hyperparameters:")
logger.info(study.best_params)



'''
hyper params to tune:
parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 8.')
parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')
parser.add_argument('--lr', type=float, default=-1, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')

# only for emp
parser.add_argument('--cont-init-std', type=float, default=0.1 , help='Initialization std for controller params. Default is 0.1.')

# only for probabilistic
parser.add_argument('--prior-std', type=float, default=7, help='Gaussian prior std. Default is 7.')
parser.add_argument('--gibbs-lambda', type=float, default=None , help='Lambda is the tempretaure of the Gibbs distribution. Default is lambda_star (see the paper).')
(8*args.num_rollouts*math.log(1/args.delta))**0.5
'''