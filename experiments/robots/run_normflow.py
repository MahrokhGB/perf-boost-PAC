import sys, os, logging, torch
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger

# NEW
import math
from tqdm import tqdm
import normflows as nf
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist import GibbsWrapperNF

BASE_IS_PRIOR = False
LEARN_BASE = True
TRAIN_METHOD = 'normflow'

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, args.cont_type+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- ' + TRAIN_METHOD + ' ----------\n\n')
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data, test_data = train_data.to(device), test_data.to(device)
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
plot_data = plot_data.to(device)
# batch the data
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # TODO
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(
    xbar=dataset.xbar, x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const
).to(device)

# ------------ 3. Controller ------------
if args.cont_type=='PerfBoost':
    ctl_generic = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        initialization_std=args.cont_init_std,
        output_amplification=20, train_method=TRAIN_METHOD
    ).to(device)
elif args.cont_type=='Affine':
    ctl_generic = AffineController(
        weight=torch.zeros(sys.in_dim, sys.state_dim, device=device, dtype=torch.float32),
        bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32),
        train_method=TRAIN_METHOD
    )
elif args.cont_type=='NN':
    ctl_generic = NNController(
        in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=[],
        train_method=TRAIN_METHOD
    )
else:
    raise KeyError('[Err] args.cont_type must be PerfBoost, NN, or Affine.')

num_params = ctl_generic.num_params
logger.info('[INFO] Controller is of type ' + args.cont_type + ' and has %i parameters.' % num_params)

# ------------ 4. Loss ------------
Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
loss_bound = 1
x0 = dataset.x0.reshape(1, -1).to(device)
sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
sat_bound += 0 if args.alpha_col is None else args.alpha_col
sat_bound += 0 if args.alpha_obst is None else args.alpha_obst
sat_bound = sat_bound/20
logger.info('Loss saturates at: '+str(sat_bound))
bounded_loss_fn = RobotsLossMultiBatch(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=loss_bound, sat_bound=sat_bound.to(device),
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)
original_loss_fn = RobotsLossMultiBatch(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)

# ------------ 5. NEW: Prior ------------
if args.cont_type in ['Affine', 'NN']:
    prior_dict = {
        'type':'Gaussian', 'type_w':'Gaussian',
        'type_b':'Gaussian_biased',
        'weight_loc':0, 'weight_scale':1,
        'bias_loc':0, 'bias_scale':5,
    }
else:
    prior_std = 7
    prior_dict = {'type':'Gaussian'}
    training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
    for name in training_param_names:
        prior_dict[name+'_loc'] = 0
        prior_dict[name+'_scale'] = prior_std

# ------------ 6. NEW: Posterior ------------
epsilon = 0.1       # PAC holds with Pr >= 1-epsilon
gibbs_lambda_star = (8*args.num_rollouts*math.log(1/epsilon))**0.5   # lambda for Gibbs
logger.info('gibbs_lambda_star %f' % gibbs_lambda_star)
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda_star, prior_dict=prior_dict,
    num_ensemble_models=40, #TODO
    # attributes of the CL system
    controller=ctl_generic, sys=sys,
    # misc
    logger=logger,
)

# Wrap Gibbs distribution to be used in normflows
target = GibbsWrapperNF(
    target_dist=gibbs_posteior, train_dataloader=train_dataloader,
    prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
)

# ****** INIT NORMFLOWS ******
num_flows = 16
from_type = 'Planar'

flows = []
for i in range(num_flows):
    if from_type == 'Radial':
        flows += [nf.flows.Radial((num_params,))]
    elif from_type == 'Planar':
        flows += [nf.flows.Planar((num_params,))]
    else:
        raise NotImplementedError

# base distribution
q0 = nf.distributions.DiagGaussian(num_params)
# base distribution same as the prior
if BASE_IS_PRIOR:
    state_dict = q0.state_dict()
    state_dict['loc'] = torch.tensor(
        [prior_dict['weight_loc'], prior_dict['bias_loc']]
    ).reshape(1, -1)
    state_dict['log_scale'] = torch.log(torch.tensor(
        [prior_dict['weight_scale'], prior_dict['bias_scale']]
    )).reshape(1, -1)
    q0.load_state_dict(state_dict)

# set up normflow
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm.to(device)  # Move model on GPU if available

msg = '\n[INFO] Norm flows setup: num transformations: %i' % num_flows
msg += ' -- flow type: ' + str(type(flows[0])) + ' -- base dist: ' + str(type(q0))
logger.info(msg)

# plot closed-loop trajectories by sampling controller from untrained nfm
logger.info('Plotting closed-loop trajectories before training the normalizing flow model.')
with torch.no_grad():
    z, _ = nfm.sample(1)
    ctl_generic.set_parameters_as_vector(z[0, :])
    x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    dataset.xbar, sys.n_agents, text="CL - before training", T=t_ext,
    save_folder=save_folder, filename='CL_init.pdf',
    obstacle_centers=bounded_loss_fn.obstacle_centers,
    obstacle_covs=bounded_loss_fn.obstacle_covs
)
# evaluate on the train data
num_samples_nf_eval = 40 #TODO
logger.info('\n[INFO] evaluating the base distribution on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=q0, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % train_loss
if args.col_av:
    msg += ' -- Average number of collisions = %i' % train_num_col
logger.info(msg)
# evaluate on the train data
logger.info('\n[INFO] evaluating the initial flow on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % train_loss
if args.col_av:
    msg += ' -- Average number of collisions = %i' % train_num_col
logger.info(msg)

# ****** TRAIN NORMFLOWS ******
# Train model
num_samples_nf_train = 400
num_samples_nf_eval = 100
anneal_iter = int(args.epochs/2)
annealing = False # NOTE
weight_decay = 0
msg = '[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
msg += ' -- learning rate: %.6f' % args.lr + ' -- weight decay: %.6f' % weight_decay
logger.info(msg)

nf_loss_hist = [None]*args.epochs

optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=weight_decay)
param_history = list(nfm.parameters()).detach().clone().numpy().reshape(1,-1)
with tqdm(range(args.epochs)) as t:
    for it in t:
        optimizer.zero_grad()
        if annealing:
            # t_now = time.time()
            nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + it / anneal_iter]))
            # print('reverse KLD time ', time.time()-t_now)
        else:
            nf_loss = nfm.reverse_kld(num_samples_nf_train)
        nf_loss.backward()
        for name, param in nfm.named_parameters():
            if not torch.isfinite(param.grad).all():
                logger.info('grad for ' + name + 'is infinite.')
            if param.isnan().any():
                logger.info(param, ' is nan in iter ', it)
        optimizer.step()

        # add to history
        param_history = np.concatenate(
            (param_history, list(nfm.parameters()).detach().clone().numpy().reshape(1,-1)),
            axis=0
        )

        nf_loss_hist[it] = nf_loss.to('cpu').data.numpy()

        # Eval and log
        if (it + 1) % args.log_epoch == 0 or it+1==args.epochs:
            # sample some controllers and eval
            with torch.no_grad():
                z, _ = nfm.sample(num_samples_nf_eval)
                z_mean = torch.mean(z, axis=0).reshape(1, -1)
                lpl = target.target_dist._log_prob_likelihood(params=z, train_data=train_data)
                lpl_mean = target.target_dist._log_prob_likelihood(params=z_mean, train_data=train_data)

            # log nf loss
            elapsed = t.format_dict['elapsed']
            elapsed_str = t.format_interval(elapsed)
            msg = 'Iter %i' % (it+1) + ' --- elapsed time: ' + elapsed_str  + ' --- norm flow loss: %f'  % nf_loss.item()
            msg += ' --- train loss %f' % torch.mean(lpl) + ' --- train loss of mean %f' % torch.mean(lpl_mean)
            logger.info(msg)

            # save nf model
            name = 'final' if it+1==args.epochs else 'itr '+str(it+1)
            torch.save(nfm.state_dict(), os.path.join(save_folder, name+'_nfm'))
            # plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(nf_loss_hist, label='loss')
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'loss.pdf'))
            plt.show()
            # plot param history
            plt.figure(figsize=(10, 10))
            for param_num in range(num_params):
                plt.plot(nf_loss_hist[:, param_num], label='param '+str(param_num))
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'param.pdf'))
            plt.show()
            # plot closed_loop
            with torch.no_grad():
                ctl_generic.set_parameters_as_vector(z_mean)
                x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
            plot_trajectories(
                x_log[0, :, :], # remove extra dim due to batching
                dataset.xbar, sys.n_agents, text="CL - "+name, T=t_ext,
                save_folder=save_folder, filename='CL_'+name+'.pdf',
                obstacle_centers=bounded_loss_fn.obstacle_centers,
                obstacle_covs=bounded_loss_fn.obstacle_covs
            )

# ------ 7. evaluate the trained model ------
# evaluate on the train data
logger.info('\n[INFO] evaluating the trained flow on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % train_loss
if args.col_av:
    msg += ' -- Average number of collisions = %i' % train_num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained flow on %i test rollouts.' % test_data.shape[0])
test_loss, test_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=test_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % test_loss
if args.col_av:
    msg += ' -- Average number of collisions = %i' % test_num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
with torch.no_grad():
    z, _ = nfm.sample(1)
    ctl_generic.set_parameters_as_vector(z[0, :])
    x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
filename = os.path.join(save_folder, 'CL_trained.pdf')
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    dataset.xbar, sys.n_agents, text="CL - trained controller", T=t_ext,
    filename='CL_trained.pdf', save_folder=save_folder,
    obstacle_centers=bounded_loss_fn.obstacle_centers,
    obstacle_covs=bounded_loss_fn.obstacle_covs
)
