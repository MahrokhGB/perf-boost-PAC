import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
from arg_parser import argument_parser, print_args
from plants import LTISystem, LTIDataset
# from plot_functions import *
from controllers import AffineController, NNController, PerfBoostController
from loss_functions import LQLossFH
from utils.assistive_functions import WrapLogger, sample_2d_dist

import math
from tqdm.auto import tqdm      # TODO train with train_normflow
import normflows as nf
from inference_algs.Gibbs2d_assistive_functions import *
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist import GibbsWrapperNF

import numpy as np
import inference_algs.normflow_assist.mynf as mynf # TODO
from control import dlqr
import matplotlib.pyplot as plt
import pickle

BASE_IS_PRIOR = False
LEARN_BASE = True
TRAIN_METHOD = 'normflow'

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
save_folder = os.path.join(save_path, args.cont_type+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- ' + TRAIN_METHOD + ' ----------\n\n')
logger.info(msg)
PLOT_DIST = True #args.cont_type=='Affine'

# torch.manual_seed(args.random_seed)

# ------ 1. load data ------
num_test_samples = 512
prior_type_b = 'Gaussian_biased_wide' # 'Uniform' #'Gaussian_biased_wide'
state_dim = 1
d_dist_v = 0.3*np.ones((state_dim, 1))
disturbance = {
    'type':'N biased',
    'mean':0.3*np.ones(state_dim),
    'cov':np.matmul(d_dist_v, np.transpose(d_dist_v))
}
dataset = LTIDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=state_dim, disturbance=disturbance
)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=num_test_samples)
train_data, test_data = train_data.to(device).float(), test_data.to(device).float()
# batch the data
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) #NOTE
train_dataloader = DataLoader(train_data, batch_size=min(args.num_rollouts, 512), shuffle=False)

# ------------ 2. Plant ------------
sys = LTISystem(
    A = torch.tensor([[0.8]]),  # state_dim*state_dim
    B = torch.tensor([[0.1]]),  # state_dim*in_dim
    C = torch.tensor([[0.3]]),  # num_outputs*state_dim
    x_init = 2*torch.ones(1, 1),# state_dim*1
).to(device)

# ------------ 3. Controller ------------
if args.cont_type=='PerfBoost':
    ctl_generic = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        initialization_std=args.cont_init_std,
        output_amplification=20,
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
Q = 5*torch.eye(sys.state_dim).to(device)
R = 0.003*torch.eye(sys.in_dim).to(device)
# optimal loss bound
loss_bound = 1
sat_bound = torch.matmul(torch.matmul(torch.transpose(sys.x_init, 0, 1), Q), sys.x_init)
if loss_bound is not None:
    logger.info('[INFO] bounding the loss to ' + str(loss_bound))
bounded_loss_fn = LQLossFH(Q, R, loss_bound, sat_bound)
original_loss_fn = LQLossFH(Q, R, None, None)

# ------------ 5. Prior ------------
# ------ prior on weight ------
# center prior at the infinite horizon LQR solution
K_lqr_ih, _, _ = dlqr(
    sys.A.detach().cpu().numpy(), sys.B.detach().cpu().numpy(),
    Q.detach().cpu().numpy(), R.detach().cpu().numpy()
)
theta_mid_grid = -K_lqr_ih[0,0]

# ------ prior on bias ------
use_ideal = True
if isinstance(ctl_generic, AffineController):
    prior_dict = {
        'type_w':'Gaussian',
        'weight_loc': theta_mid_grid if use_ideal else 0,
        'weight_scale':1 if use_ideal else 10,
    }
    if prior_type_b == 'Uniform':
        prior_dict.update({
            'type_b':'Uniform',
            'bias_low':-5, 'bias_high':5
        })
    elif prior_type_b == 'Gaussian_biased_wide':
        prior_dict.update({
            'type_b':'Gaussian_biased',
            'bias_loc': -disturbance['mean'][0]/sys.B[0,0] if use_ideal else 0,
            'bias_scale':1.5 if use_ideal else 5,
        })
elif isinstance(ctl_generic, NNController):
    prior_dict = {
        'type':'Gaussian', 'type_b':'Gaussian', 'type_w':'Gaussian',
        'weight_loc': theta_mid_grid if use_ideal else 0,
        'weight_scale':1 if use_ideal else 10,
        'bias_loc': -disturbance['mean'][0]/sys.B[0,0] if use_ideal else 0,
        'bias_scale':1.5 if use_ideal else 5,
    }
elif isinstance(ctl_generic, PerfBoostController):
    prior_std = 7
    prior_dict = {'type':'Gaussian'}
    training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
    for name in training_param_names:
        prior_dict[name+'_loc'] = 0
        prior_dict[name+'_scale'] = prior_std

# ------------ 6. NEW: Posterior ------------
gibbs_lambda_star = (8*args.num_rollouts*math.log(1/args.delta))**0.5   # lambda for Gibbs
lambda_ = gibbs_lambda_star
msg = ' -- delta: %.1f' % args.delta + ' -- prior over bias: ' + prior_type_b + ' -- lambda: %.1f' % lambda_
logger.info(msg)

# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda_star,
    prior_dict=prior_dict,
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

# load gridded Gibbs distribution
if PLOT_DIST:
    # grid
    weight_range = 4 if args.cont_type=='Affine' else 20
    theta_grid, bias_grid = get_grid(prior_dict=prior_dict, n_grid=65, weight_range=weight_range)
    theta_grid, bias_grid = theta_grid.to(device), bias_grid.to(device)
    filename = os.path.join(os.path.dirname(save_folder), 'loss_dict_S'+str(args.num_rollouts)+'.pt')
    loaded = False

    # load loss over the grid
    if os.path.isfile(filename):
        # load loss over the grid
        filehandler = open(filename, 'rb')
        loss_dict = pickle.load(filehandler)
        filehandler.close()
        # check bias and theta grid match
        if (loss_dict['theta_grid']==theta_grid).all() and (loss_dict['bias_grid']==bias_grid).all():
            loaded = True
            logger.info('Loaded the loss over the grid.')
    # compute loss over the gric
    if not loaded:
        logger.info('Computing the loss over the grid.')
        loss_dict = compute_loss_by_gridding(
            theta_grid, bias_grid, sys, train_data=train_data, ctl_generic=ctl_generic,
            loss_fn=bounded_loss_fn, save_folder=os.path.dirname(save_folder), test_data=test_data,
        )

    # compute posterior
    post_dict = compute_posterior_by_gridding(
        prior_dict=prior_dict, gibbs_lambda=gibbs_posteior.lambda_, loss_dict=loss_dict
    )
    # reshape posterior as matrix
    Z_posterior = torch.stack(post_dict['posterior'], dim=0).reshape(
        len(theta_grid), len(bias_grid)
    ).detach().cpu().numpy()
    assert abs(sum(sum(Z_posterior))-1)<=1e-5
    # convert to numpy for plotting
    theta_grid, bias_grid = theta_grid.detach().cpu().numpy(), bias_grid.detach().cpu().numpy()

# ****** INIT NORMFLOWS ******
num_flows = 1
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
q0 = nf.distributions.DiagGaussian(num_params, trainable=LEARN_BASE)
# base distribution same as the prior
std_scale = 5 # set to 1 for no scale
mean_shift = 0 # set to 0 for no shift
if BASE_IS_PRIOR:
    if isinstance(ctl_generic, AffineController):
        assert prior_type_b.startswith('Gauss')
    if LEARN_BASE:
        state_dict = q0.state_dict()
        state_dict['loc'] = gibbs_posteior.prior.mean().reshape(1, -1) + torch.Tensor([mean_shift])
        state_dict['log_scale'] = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1)) + torch.log(torch.Tensor([std_scale]))
        q0.load_state_dict(state_dict)
    else:
        q0.loc = gibbs_posteior.prior.mean().reshape(1, -1) + torch.Tensor([mean_shift])
        q0.log_scale = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1)) + torch.log(torch.Tensor([std_scale]))
    msg = '[INFO] base distribution is set using the prior, with '
    msg += 'no min shift' if mean_shift==0 else 'mean shift of %.2f.' % mean_shift
    msg += ' and same scale.' if std_scale==1 else ' %.2f times larger scale.'%std_scale
    logger.info(msg)
else:   # NOTE
    if LEARN_BASE:
        state_dict = q0.state_dict()
        state_dict['loc'] = state_dict['loc']+ torch.Tensor([mean_shift])
        state_dict['log_scale'] = state_dict['log_scale']+torch.log(torch.Tensor([std_scale]))
        q0.load_state_dict(state_dict)
    else:
        q0.loc = q0.loc + torch.Tensor([mean_shift])
        q0.log_scale = q0.log_scale + torch.log(torch.Tensor([std_scale]))
msg = '\n[INFO] Norm flows setup: num transformations: %i' % len(flows)
msg += ' -- flow type: ' + str(type(flows[0])) + ' -- base dist: ' + str(type(q0))
logger.info(msg)

# set up normflow
nfm = mynf.NormalizingFlow(q0=q0, flows=flows, p=target) # TODO
nfm.to(device)  # Move model on GPU if available

# plot before training
if PLOT_DIST:
    dist_fig, dist_axs = plt.subplots(2, 3, figsize=(15, 10))
    hist_range=[[bias_grid[-1], bias_grid[0]], [theta_grid[-1], theta_grid[0]]]
    scale_range = 1.0
    hist_range = [[0.5*((1+scale_range)*a[0]+(1-scale_range)*a[1]), 0.5*((1-scale_range)*a[0]+(1+scale_range)*a[1])]for a in hist_range]
    bins=(len(bias_grid), len(theta_grid))

    for ax in dist_axs.flatten():
        ax.set_xlabel('bias')
        ax.set_ylabel('weight')

    # plot target
    z_ind = sample_2d_dist(dist=np.transpose(Z_posterior), num_samples=2 ** 20)
    dist_axs[1,2].hist2d(
        [bias_grid[z_ind[ind, 0]] for ind in range(z_ind.shape[0])],
        [theta_grid[z_ind[ind, 1]] for ind in range(z_ind.shape[0])],
        bins=bins, range=hist_range
    )
    dist_axs[1,2].set_title('target distribution')

    # Plot initial flow distribution
    z, _ = nfm.sample(num_samples=2 ** 20)
    z = z.detach().cpu().numpy()
    dist_axs[0, 1].hist2d(
        z[:, 1].flatten(), z[:, 0].flatten(),
        bins=bins, range=hist_range
    )
    dist_axs[0, 1].set_title('initial flow distribution')

    # Plot initial base distribution
    nfm_base = nf.NormalizingFlow(q0=q0, flows=[], p=target)
    nfm_base.to(device)
    z, _ = nfm_base.sample(num_samples=2 ** 20)
    z = z.to('cpu').data.numpy()
    dist_axs[0,0].hist2d(
        z[:, 1].flatten(), z[:, 0].flatten(),
        bins=bins, range=hist_range
    )
    dist_axs[0,0].set_title('initial base distribution')

# evaluate on the train data
num_samples_nf_eval = 40 #TODO
logger.info('\n[INFO] evaluating the base distribution on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=q0, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=False
)
msg = 'Average loss: %.4f' % train_loss
logger.info(msg)
# evaluate on the train data
logger.info('\n[INFO] evaluating the initial flow on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=False
)
msg = 'Average loss: %.4f' % train_loss
logger.info(msg)


# ****** TRAIN NORMFLOWS ******
num_samples = 2 * 20
anneal_iter = int(0.75*args.epochs)
annealing = False # NOTE
msg = '\n[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
logger.info(msg)

loss_hist = []
loss_fig, loss_ax = plt.subplots(1,1,figsize=(10,10))
loss_ax.set_xlabel('iteration')
loss_ax.set_ylabel('norm flow loss ('+'annealed)' if annealing else 'not annealed)')

optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

with tqdm(range(args.epochs)) as t:
    for it in t:
        # print(list(nfm.parameters()))
        optimizer.zero_grad()
        if annealing:
            nf_loss = nfm.reverse_kld(num_samples, beta=min([1., 0.01 + it / anneal_iter]))
        else:
            nf_loss = nfm.reverse_kld(num_samples)
        nf_loss.backward()
        optimizer.step()

        loss_hist = loss_hist + [nf_loss.to('cpu').data.numpy()]
        # Plot learned distribution
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
                # save
                torch.save(nfm.state_dict(), os.path.join(save_folder, 'nfm'))

                # plot
                if PLOT_DIST:
                    # plot learned flow
                    z, _ = nfm.sample(2**20)
                    z = z.detach().cpu().numpy()
                    dist_axs[1,1].hist2d(
                        z[:, 1].flatten(), z[:, 0].flatten(),
                        bins=bins, range=hist_range
                    )
                    dist_axs[1,1].scatter(z[:50, 1].flatten(), z[:50, 0].flatten(), s=1)
                    name = 'final' if it+1==args.epochs else 'at itr '+str(it+1)
                    dist_axs[1,1].set_title('flow distribution - '+name)
                    # plot learned base dist
                    # Plot initial base distribution
                    nfm_base = nf.NormalizingFlow(q0=copy.deepcopy(nfm.q0), flows=[], p=target)
                    nfm_base.to(device)
                    z, _ = nfm_base.sample(num_samples=2 ** 20)
                    z = z.to('cpu').data.numpy()
                    dist_axs[1,0].hist2d(
                        z[:, 1].flatten(), z[:, 0].flatten(),
                        bins=bins, range=hist_range
                    )
                    dist_axs[1,0].set_title('base distribution - '+name)
                    # save fig
                    dist_fig.savefig(os.path.join(save_folder, 'distributions_'+name+'.pdf'))
                    dist_fig.show()

                # plot loss
                loss_ax.plot(loss_hist, color='k', label='loss' if (it+1)==args.log_epoch else '')
                loss_ax.plot(
                    [-x.cpu().numpy()*bet for x, bet in zip(nfm.log_p_mean_hist, nfm.beta_hist)],
                    color='b', linestyle='dashed', label='- beta * log p' if (it+1)==args.log_epoch else ''
                )
                loss_ax.plot(
                    [x.cpu().numpy() for x in nfm.log_q_mean_hist],
                    color='c', linestyle='dashed', label='log q' if (it+1)==args.log_epoch else ''
                )
                loss_ax.legend()
                loss_fig.savefig(os.path.join(save_folder, 'loss.pdf'))
                loss_fig.show()


'''
rejection sampling
prop_shift and scale

error with uniform prior: params go outside the support
'''