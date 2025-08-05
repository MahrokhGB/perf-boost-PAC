import optuna, sys, os, logging
from datetime import datetime

from arg_parser import argument_parser
from run_emp import train_emp
from run_SVGD import train_svgd
from run_normflow import train_normflow
from utils.assistive_functions import WrapLogger

def define_tunables(args):
    """Set tunable hyperparameters for the experiment."""
    # hidden_size = trial.suggest_int('hidden_size', 128, 512)
    # args.lr = trial.suggest_float('lr', args.lr/args.optuna_search_scale, args.lr*args.optuna_search_scale, log=True)
    if method=='empirical':
        if args.nn_type=='REN':
            tunables = [
                {
                    'name':'cont_init_std',
                    'nominal':args.cont_init_std,
                    'min':1e-3, 
                    'max':1, 
                    'log_scale':True
                },
                # {
                #     'name':'hidden_size',
                #     'nominal':args.hidden_size,
                #     'min':64, 
                #     'max':512, 
                #     'log_scale':False
                # }
            ]
        elif args.nn_type=='SSM':
            tunables = [
                {
                    'name':'rmin',
                    'nominal':args.rmin,
                    'min':1e-1, 
                    'max':0.99, 
                    'log_scale':True
                },
            ]
        else:
            tunables = [
                {
                    'name':'cont_init_std',
                    'nominal':args.cont_init_std,
                    'min':1e-3, 
                    'max':1, 
                    'log_scale':True
                }
            ]
    elif method in ['SVGD', 'normflow']:
        if not args.nominal_prior:
            tunables = [
                {
                    'name':'prior_std',
                    'nominal':args.prior_std,
                    'min':args.prior_std/args.optuna_search_scale, 
                    'max':args.prior_std*args.optuna_search_scale, 
                    'log_scale':True
                },
            ]
        else:
            tunables = [
                {
                    'name':'nominal_prior_std_scale',
                    'nominal':args.nominal_prior_std_scale,
                    'min':args.nominal_prior_std_scale/args.optuna_search_scale,
                    'max':args.nominal_prior_std_scale*args.optuna_search_scale,
                    'log_scale':True
                }
            ]
        # tunables +=[{
                #     'name':'gibbs_lambda',
                #     'nominal':args.gibbs_lambda,
                #     'min':args.gibbs_lambda/args.optuna_search_scale,
                #     'max':args.gibbs_lambda*args.optuna_search_scale,
                #     'log_scale':True
                # }]
        # if method=='normflow':
        #     tunables += [
                # {
                #     'name':'planar_flow_scale',
                #     'nominal':args.planar_flow_scale,
                #     'min':args.planar_flow_scale/args.optuna_search_scale,
                #     'max':args.planar_flow_scale*args.optuna_search_scale,
                #     'log_scale':True
                # },
            # ]
    else:
        raise ValueError("Method not recognized. Choose from 'empirical', 'SVGD', or 'normflow'.")
    
    return tunables

    
    
def objective(trial):
    # change tunable args to optuna format
    for tunable in tunables:
        setattr(args, tunable['name'], trial.suggest_float(
            tunable['name'], 
            tunable['min'], 
            tunable['max'], 
            log=tunable.get('log_scale', False)
        ))
    
    # get training loss
    if method=='empirical':
        res_dict, _ = train_emp(args, logger, save_folder)
    elif method=='SVGD':
        res_dict, _ = train_svgd(args, logger, save_folder)
    elif method=='normflow':
        res_dict, _ = train_normflow(args, logger, save_folder)

    return res_dict['train_loss']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

# ----- parse and set experiment arguments -----
args = argument_parser()

# ----- SET UP LOGGER -----
method = args.optuna_training_method
assert not method is None

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

# define the tunables based on the method
tunables = define_tunables(args)

# ----- START OPTUNA STUDY -----
logger.info("Starting Hyperparameter Optimization with Optuna")
study = optuna.create_study(direction='minimize')
# Define the nominal trial
nominal_trail = {}
for tunable in tunables:
    nominal_trail[tunable['name']] = tunable['nominal']
study.enqueue_trial(nominal_trail)
# Optimize the study with the objective function
study.optimize(objective, n_trials=args.optuna_n_trials)
logger.info("Best Hyperparameters:")
logger.info(study.best_params)
