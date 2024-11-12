import sys, os, logging, torch, time, math
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import normflows as nf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
# from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist.mynf import NormalizingFlow
from inference_algs.normflow_assist import GibbsWrapperNF
from ub_utils import get_mcdim_ub, get_max_lambda


delta = 0.1
num_rollouts = 128
num_prior_samples = 10**6
lambda_ = (8*num_rollouts*math.log(1/delta))**0.5       # experiments use lambda_*
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')

# ------------ 1. Basics ------------
# Dataset
dataset = RobotsDataset(
    random_seed=5, horizon=100,
    std_ini=0.2, n_agents=2
)
train_data_full, test_data = dataset.get_data(num_train_samples=num_rollouts, num_test_samples=500)
train_data_full, test_data = train_data_full.to(device), test_data.to(device)

# Plant
x0 = torch.tensor([2., -2, 0, 0, -2, -2, 0, 0,])
xbar = torch.tensor([-2, 2, 0, 0, 2., 2, 0, 0,])
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(
    xbar=xbar, x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=False, k=1.0
).to(device)

# Controller
ctl_generic = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=8, dim_nl=8,
    initialization_std=0.1,
    output_amplification=20, train_method='normflow'
).to(device)

# Loss
Q = torch.kron(torch.eye(2), torch.eye(4)).to(device)
x0 = x0.reshape(1, -1).to(device)
sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t()) + 5100
sat_bound = sat_bound/20
bounded_loss_fn = RobotsLossMultiBatch(
    Q=Q, alpha_u=0.1/400, xbar=xbar,
    loss_bound=1, sat_bound=sat_bound.to(device),
    alpha_col=100, alpha_obst=5000,
    min_dist=1.0,
    n_agents=sys.n_agents,
)
C = bounded_loss_fn.loss_bound

# ------------ 2. Range for num_rollouts_prior ------------
num_rollouts_P = np.arange(1, num_rollouts)
num_rollouts_Q = num_rollouts*np.ones(num_rollouts-1) - num_rollouts_P
lambdas_P = lambda_ * num_rollouts_P / num_rollouts
lambdas_Q = lambda_ * num_rollouts_Q / num_rollouts

# compute epsilon/lambda + ub_const for different lambda_Q
mcdim_terms = [get_mcdim_ub(
        sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
        num_prior_samples=num_prior_samples, delta=delta, C=C, deltahat=delta,
        batch_size=1000, return_keys=['epsilon/lambda_', 'ub_const'],
        lambda_=lambdas_Q[ind],                             # lambda for posterior training
        train_data=train_data_full[num_rollouts_P[ind]:],   # remove data used for training the prior
    ) for ind in range(num_rollouts-1)
]
# divide to different terms
res_eps = [mcdim_terms[ind]['epsilon/lambda_'].item() for ind in range(num_rollouts-1)]
res_const = [mcdim_terms[ind]['ub_const'].item() for ind in range(num_rollouts-1)]
res_tot = [mcdim_terms[ind]['tot'].item() for ind in range(num_rollouts-1)]
# find where the sum is smaller than thresh, where thresh is set to 1.5 * min
thresh = 1.5*min(res_tot)
inds = [i for i in range(len(res_tot)) if res_tot[i] <= thresh]
lambda_Q_range = [lambdas_Q[ind] for ind in inds]
lambda_Q_range.reverse()
print('[INFO] lambda_Q range for epsilon/lambda_Q + ub_const <= {:.2f} is {:.2f} - {:.2f}'.format(
    thresh, lambda_Q_range[0], lambda_Q_range[-1]))

# Plot
fig, ax = plt.subplots(1,1, figsize=(4,3))

y = np.vstack([[res_eps[ind] for ind in inds], [res_const[ind] for ind in inds]])
ax.stackplot(
    [lambdas_Q[ind] for ind in inds],
    y,
    labels=['epsilon/lambda_', 'ub_const']
)
ax.set_title('number of training rollouts = '+str(num_rollouts))
ax.set_xlabel('lambda')
ax.set_ylabel('upper bound')
ax.legend(loc='upper left')
# print('best lambda for S')

# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'test.png'))