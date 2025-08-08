import sys, os, logging, torch, copy, optuna
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from arg_parser import argument_parser, print_args
from ub_utils import get_mcdim_ub
from experiments.robots.run_normflow import train_normflow

# tune nominal prior std
def objective(trial):
    args_step1.nominal_prior_std_scale = trial.suggest_float(
        'nominal_prior_std_scale', 
        args_step1.nominal_prior_std_scale/10, 
        args_step1.nominal_prior_std_scale*10, 
        log=True
    )

    # train the model
    _, _, nfm = train_normflow(args_step1, logger, save_path_rob)

    # compute upper bound
    # logger.info('\nComputing the upper bound using '+str(num_prior_samples)+' prior samples.')
    ub = get_mcdim_ub(
        sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
        num_prior_samples=num_prior_samples, delta=args.delta, C=C, deltahat=args.delta,
        batch_size=1000,
        prior=nfm,                          # the trained nfm is the prior
        lambda_=lambda_Q,                   # lambda for posterior training
        train_data=train_data_full[S_P:],   # remove data used for training the prior
    )

    return ub['tot']  # return the total upper bound value




# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args, 'normflow')

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'ub_ub', 'saved_results')
save_path_rob = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, 'twostep_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- Two step training ----------\n')
logger.info(msg)
torch.manual_seed(args.random_seed)
num_prior_samples = 10**12   # TODO: move to arg parser

# ------------ 1. Basics ------------
# Dataset
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
# divide to train and test
train_data_full, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data_full, test_data = train_data_full.to(device), test_data.to(device)

# Plant
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(
    xbar=dataset.xbar, x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const
).to(device)

# Controller
assert args.cont_type=='PerfBoost'
ctl_generic = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init,
    output_init=sys.u_init,
    nn_type=args.nn_type,
    dim_internal=args.dim_internal,
    output_amplification=args.output_amplification,
    train_method='normflow',
    # SSM properties
    scaffolding_nonlin=args.scaffolding_nonlin,
    dim_middle=args.dim_middle,
    dim_scaffolding=args.dim_scaffolding,
    rmin=args.rmin,
    rmax=args.rmax,
    max_phase=args.max_phase,
    # REN properties
    dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    #   pos_def_tol=args.pos_def_tol,
    # contraction_rate_lb = args.contraction_rate_lb,
    # ren_internal_state_init=None,  # None for random initialization
).to(device)
num_params = ctl_generic.num_params
logger.info('[INFO] Controller is of type ' + args.cont_type + ' and has %i parameters.' % num_params)

# Loss
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
C = bounded_loss_fn.loss_bound

# ------------ 2. Range for num_rollouts_prior ------------
num_rollouts_P = np.arange(1, args.num_rollouts, dtype=int)
num_rollouts_Q = args.num_rollouts*np.ones(args.num_rollouts-1, dtype=int) - num_rollouts_P
lambdas_P = args.gibbs_lambda * num_rollouts_P / args.num_rollouts
lambdas_Q = args.gibbs_lambda * num_rollouts_Q / args.num_rollouts

# compute epsilon/lambda + ub_const for different lambda_Q
mcdim_terms = [get_mcdim_ub(
        sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
        num_prior_samples=num_prior_samples, delta=args.delta, C=C, deltahat=args.delta,
        batch_size=1000, return_keys=['epsilon/lambda_', 'ub_const'],
        lambda_=lambdas_Q[ind],                             # lambda for posterior training
        train_data=train_data_full[num_rollouts_P[ind]:],   # remove data used for training the prior
    ) for ind in range(args.num_rollouts-1)
]
# divide to different terms
res_eps = [mcdim_terms[ind]['epsilon/lambda_'] for ind in range(args.num_rollouts-1)]
res_const = [mcdim_terms[ind]['ub_const'] for ind in range(args.num_rollouts-1)]
res_tot = [mcdim_terms[ind]['tot'] for ind in range(args.num_rollouts-1)]
# find where the sum is smaller than thresh, where thresh is set to 1.1 * min
thresh = 1.01*min(res_tot) # 1.1*min(res_tot) #TODO
inds = [i for i in range(len(res_tot)) if res_tot[i] <= thresh]
lambda_Q_range = [lambdas_Q[ind] for ind in inds]
lambda_P_range = [lambdas_P[ind] for ind in inds]
num_rollouts_Q_range = [num_rollouts_Q[ind] for ind in inds]
num_rollouts_P_range = [num_rollouts_P[ind] for ind in inds]
logger.info('[INFO] lambda_Q range for epsilon/lambda_Q + ub_const <= {:.2f} is {:.2f} - {:.2f}, resulting from training the prior with {:.0f} - {:.0f} rollouts.'.format(
    thresh, lambda_Q_range[-1], lambda_Q_range[0], num_rollouts_P_range[0], num_rollouts_P_range[-1]))

# Plot
fig, ax = plt.subplots(1,1, figsize=(4,3))
y = np.vstack([[res_eps[ind] for ind in inds], [res_const[ind] for ind in inds]])
ax.stackplot(
    [lambdas_Q[ind] for ind in inds],
    y,
    labels=['epsilon/lambda_', 'ub_const']
)
ax.set_title('number of total rollouts = '+str(args.num_rollouts))
ax.set_xlabel('lambda for Q')
ax.set_ylabel('upper bound without Zhat')
ax.legend(loc='upper left')
# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    save_folder, 'tuning_lambda_Q.png'))
min_ind = res_tot.index(min(res_tot))
print('min_ind', min_ind, 
      '\ngibbs_lambda', args.gibbs_lambda,
      '\nres_tot', res_tot[min_ind], 
      '\nres_const', res_const[min_ind], 
      '\nres_eps', res_eps[min_ind], 
      '\nnum_rollouts_Q', num_rollouts_Q[min_ind]
    )

# ------------ 3. Train step 1 ------------
mcdim_terms = []
for lambda_P, lambda_Q, S_P in zip(lambda_P_range, lambda_Q_range, num_rollouts_P_range):
    logger.info('\n\n------ Training prior using '+str(S_P)+' rollouts ------')
    args_step1 = copy.deepcopy(args)
    args_step1.num_rollouts = S_P
    args_step1.gibbs_lambda = lambda_P
    
    # ----- tune nominal prior std with OPTUNA -----
    logger.info("Starting Hyperparameter Optimization with Optuna")
    study = optuna.create_study(direction='minimize')
    # Define the nominal trial
    nominal_trail = {}
    nominal_trail['nominal_prior_std_scale'] = args.nominal_prior_std_scale
    study.enqueue_trial(nominal_trail)
    # Optimize the study with the objective function
    study.optimize(objective, n_trials=args.optuna_n_trials)
    logger.info("Best Hyperparameters:")
    logger.info(study.best_params)
    # store the best value of the trial
    logger.info("Best value of the trial: " + str(study.best_trial.value))
    mcdim_terms.append(study.best_trial.value)  


# divide to different terms
res_eps = [mcdim_terms[ind]['epsilon/lambda_'] for ind in range(len(mcdim_terms))]
res_const = [mcdim_terms[ind]['ub_const'] for ind in range(len(mcdim_terms))]
res_tot = [mcdim_terms[ind]['tot'] for ind in range(len(mcdim_terms))]
res_neg_log_zhat_over_lambda= [mcdim_terms[ind]['neg_log_zhat_over_lambda'] for ind in range(len(mcdim_terms))]

# Plot
fig, ax = plt.subplots(1,1, figsize=(4,3))
y = np.vstack([res_eps, res_const, res_neg_log_zhat_over_lambda])
ax.stackplot(
    lambda_Q_range,
    y,
    labels=['epsilon/lambda_', 'ub_const', 'neg_log_zhat_over_lambda']
)
ax.set_title('number of total rollouts = '+str(args.num_rollouts))
ax.set_xlabel('lambda for Q')
ax.set_ylabel('upper bound')
ax.legend(loc='upper left')
# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    save_folder, 'ub.png'))
