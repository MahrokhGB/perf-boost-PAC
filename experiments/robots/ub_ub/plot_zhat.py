import sys, os, torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem, RobotsDataset
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from inference_algs.distributions import GibbsPosterior
from ub_utils import get_neg_log_zhat_over_lambda

prior_std = 3
# S = np.logspace(start=3, stop=11, num=9, dtype=int, base=2)
# S = [256]
S = np.logspace(start=3, stop=8, num=6, dtype=int, base=2)

# ------------ 1. Dataset ------------
dataset = RobotsDataset(
    random_seed=5, horizon=100,
    std_ini=0.2, n_agents=2
)

# ------------ 2. Plant ------------
x0 = torch.tensor([2., -2, 0, 0, -2, -2, 0, 0,])
xbar = torch.tensor([-2, 2, 0, 0, 2., 2, 0, 0,])
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(
    xbar=xbar, x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=False, k=1.0
).to(device)

# ------------ 3. Controller ------------
ctl_generic = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init,
    output_init=sys.u_init,
    nn_type='REN',  # TODO: add SSM support
    dim_internal=8,
    output_amplification=20,
    train_method='normflow',
    # SSM properties
    scaffolding_nonlin=None,
    dim_middle=None,
    dim_scaffolding=None,
    rmin=None,
    rmax=None,
    max_phase=None,
    # REN properties
    dim_nl=8,
    initialization_std=0.1,
    #   pos_def_tol=args.pos_def_tol,
    # contraction_rate_lb = args.contraction_rate_lb,
    # ren_internal_state_init=None,  # None for random initialization
).to(device)

# ------------ 4. Loss ------------
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

# ------------ 5. Prior ------------
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
for name in training_param_names:
    prior_dict[name+'_loc'] = 0
    prior_dict[name+'_scale'] = prior_std
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=1, # doesn't matter
    prior_dict=prior_dict,
    # attributes of the CL system
    controller=ctl_generic, sys=sys,
    # misc
    logger=None,
)
prior = gibbs_posteior.prior
num_prior_samples = 2**9
prior_samples = prior.sample(torch.Size([num_prior_samples]))

# ---------------------------------
# ------------ 6. Plot ------------
if len(S)<=6:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
else:
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

lambda_factors = np.logspace(start=-4, stop=2, num=7, base=2)
for s_ind, num_rollouts in enumerate(S):
    ax = axs.flatten()[s_ind]
    lambdas = np.linspace(0.1, 1000, num=20)
    # lambdas = [l*num_rollouts for l in lambda_factors]
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)
    values, stats = [None]*len(lambdas), [None]*len(lambdas)
    for lambda_ind, lambda_ in enumerate(lambdas):
        values[lambda_ind], stats[lambda_ind] = get_neg_log_zhat_over_lambda(sys, ctl_generic, train_data, bounded_loss_fn, prior_samples, lambda_)
    # print(stats[0]['mean'])
    ax.plot(lambdas, values)
    ax.plot(lambdas, [stats[i]['mean'] for i in range(len(lambdas))], label=r'mean $\hat{\mathcal{L}}(\boldsymbol{\theta}, \mathcal{S})$', c='k')
    ax.plot(lambdas, [stats[i]['min'] for i in range(len(lambdas))], label=r'min $\hat{\mathcal{L}}(\boldsymbol{\theta}, \mathcal{S})$', c='k')
    ax.set_title('number of rollouts = '+str(num_rollouts))
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$-1/\lambda \ln (\hat{Z})$')
    print('Computing completed for number of rollouts = '+str(num_rollouts))

# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'neg_zhat_over_lambda_'+str(prior_std)+'.png'
))
