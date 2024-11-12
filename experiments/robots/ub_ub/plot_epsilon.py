import sys, os, math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from ub_utils import get_epsilon, get_min_np, get_max_lambda

# S = 32
deltas = [0.01, 0.05, 0.1, 0.2]
# lambda_factors = np.logspace(-4, 3, num=8, base=2)
# lambdas = S*lambda_factors
lambdas = np.round(np.linspace(1, 5, num=10), decimals=2)
# lambdas = [5]

threshs = np.linspace(0.1, 0.5, 20)

# plot required N_p to have epsilon <= thresh
print('\n------ plot required N_p to have epsilon <= thresh ------')
init_conditions =np.logspace(1, len(lambdas), num=len(lambdas), base=2)
# [2**5]*int(len(lambdas)/2) + [2**10]*(len(lambdas)-int(len(lambdas)/2))
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
min_nps = np.zeros((len(deltas), len(lambdas), len(threshs)))
for delta_ind, delta in enumerate(deltas):
    for lambda_ind, lambda_ in enumerate(lambdas):
        print('delta: ', delta,', lambda: ', lambda_)
        ax = axs.flatten()[delta_ind]
        min_np = [get_min_np(thresh=thresh, delta=delta, lambda_=lambda_, init_condition=init_conditions[lambda_ind]) for thresh in threshs]
        ax.plot(threshs, min_np, label=lambda_)
        ax.set_title('delta = ' + str(delta))
        ax.set_xlabel(r'threshold on $\epsilon/\lambda$')
        ax.set_ylabel(r'min num sampled priors ($N_p$)')
        min_nps[delta_ind, lambda_ind, :] = min_np
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
fig.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'min_np.png'
))


# plot max lambda to have epsilon <= thresh
print('\n------ plot max lambda to have epsilon <= thresh ------')
n_ps = [1e6, 1e9, 1e12, 1e16, 1e20]
init_conditions = [6,9,12,16, 16]#np.log10(n_ps)
print(init_conditions)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for delta_ind, delta in enumerate(deltas):
    for n_p_ind, n_p in enumerate(n_ps):
        print('delta: ', delta,', np: ', n_p)
        ax = axs.flatten()[delta_ind]
        max_lambda = [get_max_lambda(thresh=thresh, delta=delta, n_p=n_p, init_condition=init_conditions[n_p_ind]) for thresh in threshs]
        ax.plot(threshs, max_lambda, label=n_p)
        ax.set_title('delta = ' + str(delta))
        ax.set_xlabel(r'threshold on $\epsilon/\lambda$')
        ax.set_ylabel(r'max Gibbs temperature ($\lambda$)')
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
fig.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'max_lambda.png'
))


# plot epsilon vs N_p
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for delta_ind, delta in enumerate(deltas):
    Nps = np.linspace(1, 5*np.min(min_nps[delta_ind, :, :]), num=100)
    ax = axs.flatten()[delta_ind]
    for lambda_ind, lambda_ in enumerate(lambdas):
        epsilon_over_lambda = [get_epsilon(num_prior_samples=Np, delta=delta, lambda_=lambda_)/lambda_ for Np in Nps]
        ax.plot(Nps, epsilon_over_lambda, label=lambda_)
        ax.set_title('delta = ' + str(delta))
        ax.set_xlabel(r'num sampled priors ($N_p$)')
        ax.set_ylabel(r'$\epsilon / \lambda$')
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
fig.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'epsilon.png'
))
plt.show()

# plot required N_p to have epsilon <= thresh for CDC experiments
print('\n------ plot required N_p to have epsilon <= thresh for CDC experiments ------')
S = np.logspace(2, 5, num=4, base=2)
# init_conditions =np.logspace(10, 10+4*len(S), num=len(S), base=10)
init_conditions = [10**10, 10**12, 10**15, 3*10**20]
fig, axs = plt.subplots(math.ceil(len(S)/2), 2, figsize=(12, 8))
delta_cdc = 0.1
for s_ind, num_rollouts in enumerate(S):
    # init_conditions = [i/4**delta_ind for i in init_conditions]
    lambda_star = (8*num_rollouts*math.log(1/delta_cdc))**0.5
    print('delta: ', delta_cdc,', lambda: ', lambda_star, ', num_rollouts: ', num_rollouts)
    ax = axs.flatten()[s_ind]
    min_np = np.zeros(len(threshs))
    init_condition=init_conditions[s_ind]
    for thresh_ind, thresh in enumerate(threshs):
        min_np[thresh_ind] = get_min_np(thresh=thresh, delta=delta_cdc, lambda_=lambda_star, init_condition=init_condition)
        init_condition = min_np[thresh_ind]
        print(min_np[thresh_ind])
    ax.plot(threshs, min_np)
    ax.set_title('lambda star = {:.2f} for S = '.format(lambda_star)+str(num_rollouts))
    ax.set_xlabel(r'threshold on $\epsilon/\lambda$')
    ax.set_ylabel(r'min num sampled priors ($N_p$)')
plt.tight_layout()
fig.savefig(os.path.join(
    BASE_DIR, 'experiments', 'robots', 'saved_results', 'min_np_CDC.png'
))