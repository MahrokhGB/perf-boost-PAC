import sys, os, logging, torch, math
from datetime import datetime
from torch.utils.data import DataLoader
import normflows as nf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from arg_parser import argument_parser, print_args
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist.mynf import NormalizingFlow
from inference_algs.normflow_assist import GibbsWrapperNF
from ub_utils import get_mcdim_ub
from inference_algs.normflow_assist import fit_norm_flow

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
# usual setup
BASE_STD_SCALE = 1/args.nominal_prior_std_scale/100 # TODO
num_samples_nf_train = 100
num_samples_nf_eval = 100 # TODO

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
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=20, train_method='normflow'
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
print('args.num_rollouts', args.num_rollouts)
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
# prior for step 1
if args.data_dep_prior:
    if args.dim_nl==8 and args.dim_internal==8:
        if args.num_rollouts_prior==5:
            filename_load = os.path.join(save_path_rob, 'empirical', 'pretrained', 'trained_controller.pt')
            res_dict_loaded = torch.load(filename_load)
if args.nominal_prior:
    res_dict_loaded = []
    if args.dim_nl==8 and args.dim_internal==8:
        for _, dirs, _ in os.walk(os.path.join(save_path_rob, 'nominal')):
            for dir in dirs:
                filename_load = os.path.join(save_path_rob, 'nominal', dir, 'trained_controller.pt')
                res_dict_loaded.append(torch.load(filename_load))
    logger.info('[INFO] Loaded '+str(len(res_dict_loaded))+' nominal controllers.')
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
for name in training_param_names:
    if args.data_dep_prior:
        prior_dict[name+'_loc'] = res_dict_loaded[name]
        prior_dict[name+'_scale'] = args.prior_std
    elif args.nominal_prior:
        PRIOR_STDP_SCALE = 50 # TODO
        logger.info('[INFO] Prior distribution is the distribution over nominal controllers, with std scaled by %.4f.' % PRIOR_STDP_SCALE)
        vals = torch.stack([res[name] for res in res_dict_loaded], dim=0)
        # val and std computed elementwise. same shape as the training param
        prior_dict[name+'_loc'] = vals.mean(dim=0)  
        prior_dict[name+'_scale'] = vals.std(dim=0, correction=1) * PRIOR_STDP_SCALE
    else:
        prior_dict[name+'_loc'] = 0
        prior_dict[name+'_scale'] = args.prior_std

# posteiror for step 1
mcdim_terms = []
for lambda_P, lambda_Q, S_P in zip(lambda_P_range, lambda_Q_range, num_rollouts_P_range):
    logger.info('\n\n------ Training prior using '+str(S_P)+' rollouts ------')
    ctl_generic.reset()
    ctl_generic.c_ren.hard_reset()
    # train data for prior
    train_data_P = train_data_full[:S_P, :]
    train_dataloader_P = DataLoader(train_data_P, batch_size=min(int(S_P), 256), shuffle=False)
    # define target distribution
    gibbs_posteior = GibbsPosterior(
        loss_fn=bounded_loss_fn, lambda_=lambda_P,
        prior_dict=prior_dict, controller=ctl_generic,
        sys=sys, logger=logger,
    )
    # Wrap Gibbs distribution to be used in normflows
    target = GibbsWrapperNF(
        target_dist=gibbs_posteior, train_dataloader=train_dataloader_P,
        prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
    )
    # ------------ Define the nfm model ------------
    # flows
    flows = []
    for i in range(args.num_flows):
        if args.flow_type == 'Radial':
            flows += [nf.flows.Radial((num_params,), act=args.flow_activation)]
        elif args.flow_type == 'Planar': # f(z) = z + u * h(w * z + b)
            '''
            Default values:
                - u: uniform(-sqrt(2), sqrt(2))
                - w: uniform(-sqrt(2/num_prams), sqrt(2/num_prams))
                - b: 0
                - h: args.flow_activation (tanh or leaky_relu)
            '''
            flows += [nf.flows.Planar((num_params,), u=args.planar_flow_scale*(2*torch.rand(num_params)-1), act=args.flow_activation)]
        elif args.flow_type == 'NVP':
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([math.ceil(num_params/2), 64, 64, num_params], init_zeros=True, act=args.flow_activation)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))
        else:
            raise NotImplementedError

    # base distribution
    q0 = nf.distributions.DiagGaussian(num_params, trainable=args.learn_base)
    # base distribution same as the prior
    if args.base_is_prior:
        logger.info('[INFO] Base distribution is similar to the prior, with std scaled by %.4f.' % BASE_STD_SCALE)
        state_dict = q0.state_dict()
        state_dict['loc'] = gibbs_posteior.prior.mean().reshape(1, -1)
        state_dict['log_scale'] = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1)*BASE_STD_SCALE) 
        q0.load_state_dict(state_dict)
    # base distribution centered at the empirical or nominal controller
    elif args.base_center_emp or args.base_center_nominal:
        # get filename to load
        if args.base_center_emp:
            if args.dim_nl==1 and args.dim_internal==1:
                filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_08_29_14_58_13', 'trained_controller.pt')
            elif args.dim_nl==2 and args.dim_internal==4:
                filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_08_29_14_57_38', 'trained_controller.pt')
            elif args.dim_nl==8 and args.dim_internal==8:
                # # empirical controller avoids collisions
                # filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_01_14_16_26_11', 'trained_controller.pt')
                # empirical controller does not avoid collisions
                filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_01_19_11_31_25', 'trained_controller.pt')
        if args.base_center_nominal:
            if args.dim_nl==8 and args.dim_internal==8:
                filename_load = os.path.join(save_path_rob, 'nominal', 'PerfBoost_01_22_15_25_52', 'trained_controller.pt')
        # load the controller
        res_dict_loaded = torch.load(filename_load)
        # set the mean of the base distribution to the controller
        mean = np.array([])
        for name in training_param_names:
            mean = np.append(mean, res_dict_loaded[name].cpu().detach().numpy().flatten())
        state_dict = q0.state_dict()
        state_dict['loc'] = torch.Tensor(mean.reshape(1, -1))
        # set the scale of the base distribution
        # state_dict['log_scale'] = state_dict['log_scale'] - 100 # TODO
        state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*BASE_STD_SCALE)
        q0.load_state_dict(state_dict)
    else:
        state_dict = q0.state_dict()
        state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*BASE_STD_SCALE)
        q0.load_state_dict(state_dict)

    # set up normflow
    nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
    nfm.to(device)  # Move model on GPU if available

    # train nfm
    optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fit_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, logger=logger, loss_fn=bounded_loss_fn,
        save_folder=save_folder, train_data_full=train_data_P, test_data=test_data, plot_data=plot_data,
        return_best=args.return_best, validation_frac=args.validation_frac,
        early_stopping=args.early_stopping, n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage,
        optimizer=optimizer, epochs=args.epochs, log_epoch=args.log_epoch, annealing=args.annealing,
        anneal_iter=args.anneal_iter, num_samples_nf_train=num_samples_nf_train, num_samples_nf_eval=num_samples_nf_eval,
    )

    # compute upper bound
    logger.info('\nComputing the upper bound using '+str(num_prior_samples)+' prior samples.')
    ub = get_mcdim_ub(
        sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
        num_prior_samples=num_prior_samples, delta=args.delta, C=C, deltahat=args.delta,
        batch_size=1000,
        prior=nfm,                          # the trained nfm is the prior
        lambda_=lambda_Q,                   # lambda for posterior training
        train_data=train_data_full[S_P:],   # remove data used for training the prior
    )
    logger.info(ub)
    mcdim_terms.append(ub)


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
