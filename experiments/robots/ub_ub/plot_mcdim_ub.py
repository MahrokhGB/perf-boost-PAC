import sys, os, torch, math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
from plants import RobotsSystem, RobotsDataset
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from inference_algs.distributions import GibbsPosterior

def get_mcdim_ub(sys, ctl_generic, train_data, bounded_loss_fn, prior, delta, lambda_, C, num_prior_samples=5000, deltahat=None):
    deltahat = delta if deltahat is None else deltahat
    n_p_min = math.ceil(
        (1-math.exp(-lambda_*C)/lambda_/C)**2 * math.log(1/deltahat) / 2
    )
    assert num_prior_samples>n_p_min

    ctl_generic.reset()
    ctl_generic.c_ren.hard_reset()

    prior_samples = prior.sample(torch.Size([num_prior_samples]))
    train_loss_prior_samples = torch.zeros(num_prior_samples)
    for r in range(math.ceil(num_prior_samples/1000)):
        end_ind = min((r+1)*1000, num_prior_samples)
        tmp, _ = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=train_data,
            num_samples=None,nfm=None,
            params=prior_samples[r*1000:end_ind],
            loss_fn=bounded_loss_fn,
            count_collisions=True, return_av=False
        )
        train_loss_prior_samples[r*1000:end_ind] = tmp
    assert end_ind==num_prior_samples
    ub_const = 1/lambda_*math.log(1/delta) + lambda_*C**2/8/num_rollouts
    mean_loss = min(train_loss_prior_samples)#/num_prior_samples
    Z_hat_norm = sum(torch.exp(-lambda_*(train_loss_prior_samples-mean_loss)))/num_prior_samples
    epsilon = (math.log(1/deltahat)*num_prior_samples/2)**0.5 * math.log(1+(math.exp(lambda_*C)-1)/num_prior_samples)
    # epsilon = (1-math.exp(-lambda_*C)) * (math.log(1/deltahat)/num_prior_samples/2)**0.5
    # epsilon = (math.log(1/deltahat)*num_prior_samples/2)**0.5 * math.log((1-math.exp(-lambda_*C))/num_prior_samples)
    # epsilon = (math.exp(lambda_*C)-1) * (math.log(1/deltahat)/num_prior_samples/2)**0.5
    print('1/lambda_*epsilon', 1/lambda_*epsilon)
    mcdim_ub = ub_const - 1/lambda_*math.log(Z_hat_norm) + mean_loss + 1/lambda_*epsilon

    return mcdim_ub


delta = 0.01
prior_std = 7
# S = np.logspace(start=3, stop=11, num=9, dtype=int, base=2)
S = [256]
# S = np.logspace(start=3, stop=8, num=6, dtype=int, base=2)

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
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=8, dim_nl=8,
    initialization_std=0.1,
    output_amplification=20, train_method='normflow'
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

# ------------ 6. Prior ------------
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

mcdim_ubs = []
for num_rollouts in S:
    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(
        random_seed=5, horizon=100,
        std_ini=0.2, n_agents=2
    )
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)


    lambda_factors = [1/16, 1/8]#np.logspace(start=-4, stop=2, num=7, base=2)
    deltas = [0.01] #[0.01, 0.05, 0.1, 0.2]
    mcdim_ub = np.zeros((len(lambda_factors), len(deltas)))
    for lambda_ind, lambda_factor in enumerate(lambda_factors):
        for delta_ind, delta in enumerate(deltas):
            mcdim_ub[lambda_ind, delta_ind] = get_mcdim_ub(
                sys=sys, ctl_generic=ctl_generic, train_data=train_data, bounded_loss_fn=bounded_loss_fn,
                prior=prior, delta=delta, lambda_=lambda_factor*num_rollouts,
                C=1, num_prior_samples=50000, deltahat=delta
            )
            print(lambda_factor*num_rollouts, mcdim_ub[lambda_ind, delta_ind])
    mcdim_ubs += [mcdim_ub]
    print('computing completed for '+str(num_rollouts))


if len(S)<=6:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
else:
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
v_min = min([np.min(arr) for arr in mcdim_ubs])
v_max = 1
# v_max = max([np.max(arr) for arr in mcdim_ubs])
for ax_ind, ax in enumerate(axs.flatten()):
    sns.heatmap(
        mcdim_ubs[ax_ind], ax=ax, vmin=v_min, vmax=v_max, annot=True, cmap='mako_r',
        xticklabels=deltas, yticklabels=lambda_factors*S[ax_ind]
    )
    ax.set_title('number of rollouts = '+str(S[ax_ind]))
    ax.set_xlabel('delta')
    ax.set_ylabel('lambda')
    if ax_ind==len(S)-1:
        break

# save the image
plt.tight_layout()
plt.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'ub'+str(prior_std)+'.png'
))
