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
from inference_algs.normflow_assist import train_norm_flow


num_prior_samples = 10**6   # TODO: move to arg parser
# usual setup
U_SCALE = 0.1
STD_SCALE = 0.1 # TODO double check
num_samples_nf_train = 100
num_samples_nf_eval = 100

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args, 'normflow')

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'ub_ub', 'saved_results')
save_path_rob = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, 'datadep_prior_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- Data Dependent Prior ----------\n')
logger.info(msg)
torch.manual_seed(args.random_seed)

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
loss_bound = 1
x0 = dataset.x0.reshape(1, -1).to(device)
sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
sat_bound += 0 if args.alpha_col is None else args.alpha_col
sat_bound += 0 if args.alpha_obst is None else args.alpha_obst
sat_bound = sat_bound/20
logger.info('Loss saturates at: '+str(sat_bound.item()))
bounded_loss_fn = RobotsLossMultiBatch(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=loss_bound, sat_bound=sat_bound.to(device),
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)
C = bounded_loss_fn.loss_bound

# ------------ 2. Train the prior using the entire train data ------------
lambda_ = (8*args.num_rollouts*math.log(1/args.delta))**0.5 # experiments use lambda_*
lambda_P = lambda_ - args.num_rollouts**0.5                 # to maintain the same Q
lambda_Q = args.num_rollouts**0.5

# data-independent prior for step 1
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
for name in training_param_names:
    prior_dict[name+'_loc'] = 0
    prior_dict[name+'_scale'] = args.prior_std

# define target distribution
train_dataloader_full = DataLoader(train_data_full, batch_size=min(args.num_rollouts, 256), shuffle=False)
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=lambda_P,
    prior_dict=prior_dict, controller=ctl_generic,
    sys=sys, logger=logger,
)
# Wrap Gibbs distribution to be used in normflows
target = GibbsWrapperNF(
    target_dist=gibbs_posteior, train_dataloader=train_dataloader_full,
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
        flows += [nf.flows.Planar((num_params,), u=U_SCALE*(2*torch.rand(num_params)-1), act=args.flow_activation)]
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
    state_dict = q0.state_dict()
    state_dict['loc'] = gibbs_posteior.prior.mean().reshape(1, -1)
    state_dict['log_scale'] = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1))
    q0.load_state_dict(state_dict)
# base distribution centered at the empirical controller
elif args.base_center_emp:
    if args.dim_nl==1 and args.dim_internal==1:
        filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_08_29_14_58_13', 'trained_controller.pt')
    elif args.dim_nl==2 and args.dim_internal==4:
        filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_08_29_14_57_38', 'trained_controller.pt')
    elif args.dim_nl==8 and args.dim_internal==8:
        # empirical controller avoids collisions
        # filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_10_10_09_56_16', 'trained_controller.pt')
        # empirical controller does not avoid collisions
        filename_load = os.path.join(save_path_rob, 'empirical', 'PerfBoost_10_11_10_41_10', 'trained_controller.pt')
    res_dict_loaded = torch.load(filename_load)
    mean = np.array([])
    for name in training_param_names:
        mean = np.append(mean, res_dict_loaded[name].cpu().detach().numpy().flatten())
    state_dict = q0.state_dict()
    state_dict['loc'] = torch.Tensor(mean.reshape(1, -1))
    # state_dict['log_scale'] = state_dict['log_scale'] - 100 # TODO
    state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO
    q0.load_state_dict(state_dict)
# default base
else:
    state_dict = q0.state_dict()
    state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO

# set up normflow
nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
nfm.to(device)  # Move model on GPU if available

# train nfm
optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, logger=logger, bounded_loss_fn=bounded_loss_fn,
    save_folder=save_folder, train_data=train_data_full, test_data=test_data, plot_data=plot_data,
    optimizer=optimizer, epochs=args.epochs, log_epoch=args.log_epoch, annealing=args.annealing,
    anneal_iter=args.anneal_iter, num_samples_nf_train=num_samples_nf_train, num_samples_nf_eval=num_samples_nf_eval,
)

# compute upper bound
logger.info('\nComputing the upper bound using '+str(num_prior_samples)+' prior samples.')
ub = get_mcdim_ub(
    sys=sys, ctl_generic=ctl_generic, bounded_loss_fn=bounded_loss_fn,
    num_prior_samples=num_prior_samples, delta=args.delta, C=C, deltahat=args.delta,
    batch_size=1000,
    prior=nfm,                  # the trained nfm is the prior
    lambda_=lambda_Q,           # lambda for posterior training
    train_data=train_data_full, # posterior is trained on the entire train data
)
logger.info(ub)
