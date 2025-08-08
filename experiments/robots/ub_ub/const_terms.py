import sys, os, logging, torch
from datetime import datetime
import pandas as pd

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

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args, 'normflow')

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'ub_ub', 'saved_results')
save_path_rob = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, 'const_terms_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- Upper bound constants ----------\n')
logger.info(msg)
torch.manual_seed(args.random_seed)
num_prior_samples = 10**12   # TODO: move to arg parser

# ------------ 1. Basics ------------
# Dataset
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
# divide to train and test
train_data_full, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data_full, test_data = train_data_full.to(device), test_data.to(device)
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data_full.shape[-1], device=device)
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
plot_data = plot_data.to(device)

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
# lambda_range = np.linspace(args.gibbs_lambda/10, args.gibbs_lambda*10, 20)
lambda_range = np.logspace(
    np.log(args.gibbs_lambda/20), min(np.log(args.gibbs_lambda*10), np.log(500)), 
    50, base=np.e
)
lambda_range = np.round(lambda_range, decimals=4)
mcdim_terms = dict.fromkeys(
    ['gibbs_lambda', 'num_rollouts_Q', 'lambda_Q', 'epsilon/lambda_', 'ub_const', 'tot_const'],
)
for key in mcdim_terms.keys():
    mcdim_terms[key] = [None]*len(lambda_range)*(args.num_rollouts-1)
ind = 0

for gibbs_lambda in lambda_range:
    num_rollouts_P = np.arange(1, args.num_rollouts, dtype=int)
    num_rollouts_Q = args.num_rollouts*np.ones(args.num_rollouts-1, dtype=int) - num_rollouts_P
    lambdas_P = gibbs_lambda * num_rollouts_P / args.num_rollouts
    lambdas_Q = gibbs_lambda * num_rollouts_Q / args.num_rollouts

    # compute epsilon/lambda + ub_const for different lambda_Q
    for Q_ind in range(len(num_rollouts_Q)):
        mcdim_terms['gibbs_lambda'][ind] = gibbs_lambda
        mcdim_terms['lambda_Q'][ind] = lambdas_Q[Q_ind]
        mcdim_terms['num_rollouts_Q'][ind] = num_rollouts_Q[Q_ind]
        mcdim_ub = get_mcdim_ub(
            sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
            num_prior_samples=num_prior_samples, delta=args.delta, C=C, deltahat=args.delta,
            batch_size=1000, return_keys=['epsilon/lambda_', 'ub_const'],
            lambda_=lambdas_Q[Q_ind],                  # lambda for posterior training
            train_data=train_data_full[num_rollouts_P[Q_ind]:],   # remove data used for training the prior
        )
        mcdim_terms['epsilon/lambda_'][ind] = mcdim_ub['epsilon/lambda_']
        mcdim_terms['ub_const'][ind] = mcdim_ub['ub_const']
        mcdim_terms['tot_const'][ind] = mcdim_ub['tot']
        ind += 1
assert ind == len(lambda_range)*(args.num_rollouts-1)

# ------ Plot ------
fig, axs = plt.subplots(2, 3, figsize=(4*3, 4*2))
for ind, key in enumerate(['epsilon/lambda_', 'ub_const', 'tot_const']):
    tmp = pd.DataFrame(mcdim_terms)
    tmp = tmp.pivot(columns='num_rollouts_Q', index='gibbs_lambda', values=key)
    # 2D heat map
    sns.heatmap(tmp, ax=axs[0, ind], vmax=1)
    axs[0, ind].set_title(key)
    # plot vs lambda_Q
    axs[1,ind].scatter(mcdim_terms['lambda_Q'], [min(i, 1) for i in mcdim_terms[key]])
    axs[1,ind].set_xlabel('lambda_Q')
    axs[1,ind].set_ylabel(key)

# y = np.vstack([[res_eps[ind] for ind in inds], [res_const[ind] for ind in inds]])
# ax.stackplot(
#     [lambdas_Q[ind] for ind in inds],
#     y,
#     labels=['epsilon/lambda_', 'ub_const']
# )
# ax.set_title('number of total rollouts = '+str(args.num_rollouts))
# ax.set_xlabel('lambda for Q')
# ax.set_ylabel('upper bound without Zhat')
# ax.legend(loc='upper left')
# save the image

plt.tight_layout()
plt.savefig(os.path.join(
    save_folder, 'const_terms.png'))


min_ind = mcdim_terms['tot_const'].index(min(mcdim_terms['tot_const']))
logger.info(
    '\n\nbest pair:' +
    '\ngibbs_lambda = ' + str(mcdim_terms['gibbs_lambda'][min_ind])+ 
    '\nnum_rollouts_Q = ' + str(mcdim_terms['num_rollouts_Q'][min_ind])+ 
    '\nlambda_Q = ' + str(mcdim_terms['lambda_Q'][min_ind])+
    '\nepsilon/lambda_ = ' + str(mcdim_terms['epsilon/lambda_'][min_ind])+
    '\nub_const = ' + str(mcdim_terms['ub_const'][min_ind])+
    '\ntot_const = ' + str(mcdim_terms['tot_const'][min_ind])
)


filtered_pairs = []
closest_to_lambda_star = (1e6, None, None)
thresh = mcdim_terms['tot_const'][min_ind]*1.01
logger.info('\n\nfiltered pairs')
for ind in range(len(mcdim_terms['tot_const'])):
    if mcdim_terms['tot_const'][ind] <= thresh:
        filtered_pairs += (
            mcdim_terms['gibbs_lambda'][ind], 
            mcdim_terms['num_rollouts_Q'][ind], 
            mcdim_terms['tot_const'][ind]
        )
        if abs(args.gibbs_lambda-closest_to_lambda_star[0]) > abs(args.gibbs_lambda-mcdim_terms['gibbs_lambda'][ind]):
            closest_to_lambda_star = (
                mcdim_terms['gibbs_lambda'][ind], 
                mcdim_terms['num_rollouts_Q'][ind], 
                mcdim_terms['tot_const'][ind]
            )
        logger.info('\n'+
            'gibbs_lambda = ' + str(mcdim_terms['gibbs_lambda'][ind])+ 
            ', num_rollouts_Q = ' + str(mcdim_terms['num_rollouts_Q'][ind])+ 
            ', tot_const = ' + str(mcdim_terms['tot_const'][ind])
        )

logger.info('\n\nthe pair with the closest lambda to lambda_* is ')
logger.info(closest_to_lambda_star)