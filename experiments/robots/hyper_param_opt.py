import optuna, sys, os, logging
from datetime import datetime

from arg_parser import argument_parser
from run_emp import train_emp
from run_SVGD import train_svgd
from run_normflow import train_normflow
from utils.assistive_functions import WrapLogger

def objective(trial):
    # ----- parse and set experiment arguments -----
    args = argument_parser()

    # Hyperparameters to tune
    # hidden_size = trial.suggest_int('hidden_size', 128, 512)
    # args.lr = trial.suggest_float('lr', args.lr/10, args.lr*10, log=True)
    
    if method=='empirical':
        args.cont_init_std = trial.suggest_float('cont_init_std', 1e-3, 1, log=True)
        res_dict, _ = train_emp(args, logger, save_folder)
    elif method=='SVGD':
        args.prior_std = trial.suggest_float(
            'prior_std', args.prior_std/10, args.prior_std*10, log=True
        )
        args.gibbs_lambda = trial.suggest_float(
            'gibbs_lambda', args.gibbs_lambda/10, args.gibbs_lambda*10, log=True
        )
        res_dict, _ = train_svgd(args, logger, save_folder)
    elif method=='normflow':
        if args.max_gibbs_lambda:
            args.planar_flow_scale = trial.suggest_float(
                'planar_flow_scale', args.planar_flow_scale/10, args.planar_flow_scale*10, log=True
            )
            args.nominal_prior_std_scale = trial.suggest_float(
                'nominal_prior_std_scale', args.nominal_prior_std_scale/10, args.nominal_prior_std_scale*10, log=True
            )
            args.prior_std = trial.suggest_float(
                'prior_std', args.prior_std/10, args.prior_std*10, log=True
            )
        else: 
            args.planar_flow_scale = trial.suggest_float(
                'planar_flow_scale', args.planar_flow_scale/10, args.planar_flow_scale*10, log=True
            )
            args.nominal_prior_std_scale = trial.suggest_float(
                'nominal_prior_std_scale', args.nominal_prior_std_scale/10, args.nominal_prior_std_scale*10, log=True
            )
            # args.prior_std = trial.suggest_float(
            #     'prior_std', args.prior_std/10, args.prior_std*10, log=True
            # )
            # args.gibbs_lambda = trial.suggest_float(
            #     'gibbs_lambda', args.gibbs_lambda/10, args.gibbs_lambda*10, log=True
            # )
            # use best results from SVGD
            args.prior_std = 2.8257495793293894
            args.gibbs_lambda = 9.515015854816593

        res_dict, _ = train_normflow(args, logger, save_folder)

    return res_dict['train_loss']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

# ----- SET UP LOGGER -----
method = 'SVGD'
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
