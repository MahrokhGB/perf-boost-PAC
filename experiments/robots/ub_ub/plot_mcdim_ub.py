import sys, os, torch, math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem, RobotsDataset
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from inference_algs.distributions import GibbsPosterior
from ub_utils import get_mcdim_ub, get_max_lambda, get_min_np


delta = 0.1
prior_std = 0.00001
data_dep_prior = True
num_rollouts_prior = 5
# S = np.logspace(start=3, stop=11, num=9, dtype=int, base=2)
S = [64]
# S = np.logspace(start=6, stop=11, num=6, dtype=int, base=2)
print('mcdim_ub'+str(prior_std)+'.png')
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')

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
C = bounded_loss_fn.loss_bound

# ------------ 6. Prior ------------
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
if data_dep_prior:
    if num_rollouts_prior==5:
        filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_11_10_15_46_03', 'trained_controller.pt')
        res_dict_loaded = torch.load(filename_load)
        # prior std doesn't matter, b.c. loads only the prior mean
        # if prior_std==0.1:
        #     fname = 'PerfBoost_11_10_19_14_07'
        # elif prior_std==0.001:      # 1e-3
        #     fname = 'PerfBoost_11_10_19_14_27'
        # elif prior_std==0.00001:    # 1e-5
        #     fname = 'PerfBoost_11_10_19_14_35'
        # elif prior_std==0.00000001: # 1e-8
        #     fname = 'PerfBoost_11_10_19_14_43'

for name in training_param_names:
    if data_dep_prior:
        prior_dict[name+'_loc'] = res_dict_loaded[name]
    else:
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
# num_prior_samples = 2**800


thresh_ub_const = 0.4
thresh_eps_lambda = 0.2

mcdim_ubs = [None]*len(S)
lambdas = [None]*len(S)
num_lambdas = 4
for s_ind, num_rollouts in enumerate(S):
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)
    # lambda range to have ub const < thresh
    assert thresh_ub_const**2 >= math.log(1/delta)/2/num_rollouts
    tmp = (thresh_ub_const**2 - math.log(1/delta)/2/num_rollouts)**0.5
    lambda_min_ub_const = 4*num_rollouts/C*(thresh_ub_const-tmp)
    lambda_max_ub_const = 4*num_rollouts/C*(thresh_ub_const+tmp)
    print('lambda range to have ub const <= ', thresh_ub_const, ' is ', lambda_min_ub_const, ' - ', lambda_max_ub_const)
    # num_prior_samples to have eps/lambda_min < 1
    num_prior_samples = 10**6
    # num_prior_samples = get_min_np(thresh=0.2, delta=delta, lambda_=lambda_min, init_condition=100000, loss_bound=1, constrained=True, max_tries=100)
    print('num_prior_samples', num_prior_samples)
    # lamda max to have epsilon/lambda=1
    lambda_max_eps = get_max_lambda(thresh=thresh_eps_lambda, delta=delta, n_p=num_prior_samples, init_condition=20, loss_bound=C, constrained=True)
    print('lambda max to have eps/lambda <= ', thresh_eps_lambda, ' is ', lambda_max_eps)
    if lambda_min_ub_const > lambda_max_eps:
        mcdim_ubs[s_ind] = None
        print('[Err] lambda_min = ' + str(lambda_min_ub_const) + ' > lambda_max = ' + str(lambda_max_eps) + ' for '+str(num_rollouts))
        continue
    # lambda range
    lambdas[s_ind] = np.linspace(lambda_min_ub_const, min(lambda_max_eps, lambda_max_ub_const), num=num_lambdas)
    # compute ub
    mcdim_ubs[s_ind] = [None]*num_lambdas
    for lambda_ind, lambda_ in enumerate(lambdas[s_ind]):
        mcdim_ubs[s_ind][lambda_ind] = get_mcdim_ub(
            sys=sys, ctl_generic=ctl_generic, train_data=train_data, bounded_loss_fn=bounded_loss_fn,
            prior=prior, num_prior_samples=num_prior_samples, delta=delta, lambda_=lambda_,
            C=1, deltahat=delta
        )
    print('computing completed for '+str(num_rollouts))


if len(S)<=6:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
else:
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharey=True)

# Plot
for ind, num_rollouts in enumerate(S):
    if lambdas[ind] is None:
        continue
    ax = axs.flatten()[ind]
    res_eps = [mcdim_ubs[ind][lambda_ind]['epsilon/lambda_'].item() for lambda_ind in range(num_lambdas)]
    res_Z = [mcdim_ubs[ind][lambda_ind]['neg_log_zhat_over_lambda'].item() for lambda_ind in range(num_lambdas)]
    res_const = [mcdim_ubs[ind][lambda_ind]['ub_const'].item() for lambda_ind in range(num_lambdas)]
    res_tot = [mcdim_ubs[ind][lambda_ind]['tot'].item() for lambda_ind in range(num_lambdas)]
    y = np.vstack([res_eps, res_Z, res_const])
    ax.stackplot(
        lambdas[ind],
        y,
        labels=['epsilon/lambda_', 'neg_log_zhat_over_lambda', 'ub_const']
    )
    ax.set_title('number of training rollouts = '+str(num_rollouts))
    ax.set_xlabel('lambda')
    ax.set_ylabel('upper bound')
    ax.legend(loc='upper left')
    # print('best lambda for S')

# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'mcdim_ub'+str(prior_std)+'.png'
))
