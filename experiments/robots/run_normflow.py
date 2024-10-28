import sys, os, logging, torch, time
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
from inference_algs.distributions import GibbsPosterior

# NEW
import math
from tqdm import tqdm
from inference_algs.normflow_assist.mynf import NormalizingFlow
import normflows as nf
from inference_algs.normflow_assist import GibbsWrapperNF

TRAIN_METHOD = 'normflow'

U_SCALE = 0.1
STD_SCALE = 1 #0.1 TODO

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, TRAIN_METHOD, args.cont_type+'_'+now)
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
# NOTE: for normflow, must use all the data at each iter b.c. otherwise, the target distribution changes.
train_dataloader = DataLoader(train_data, batch_size=min(args.num_rollouts, 256), shuffle=False)

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
        in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=args.layer_sizes,
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

# ------------ 5. Prior ------------
if args.cont_type in ['Affine', 'NN']:
    training_param_names = ['weight', 'bias']
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

# ------------ 6. Posterior ------------
gibbs_lambda_star = (8*args.num_rollouts*math.log(1/args.delta))**0.5   # lambda for Gibbs
gibbs_lambda = 1 # gibbs_lambda_star # 1000 # TODO
logger.info('gibbs_lambda: %.2f' % gibbs_lambda + ' (use lambda_*)' if gibbs_lambda == gibbs_lambda_star else '')
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda, prior_dict=prior_dict,
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

# ------------ 7. save setup ------------
setup_dict = vars(args)
setup_dict['sat_bound'] = sat_bound
setup_dict['loss_bound'] = loss_bound
setup_dict['prior_dict'] = prior_dict
setup_dict['gibbs_lambda'] = gibbs_lambda
torch.save(setup_dict, os.path.join(save_folder, 'setup'))

# ------------ 8. NormFlows ------------
num_samples_nf_train = 100
num_samples_nf_eval = 100 # TODO

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
    state_dict['log_scale'] = gibbs_posteior.prior.stddev().reshape(1, -1)
    q0.load_state_dict(state_dict)
# base distribution same as the prior
elif args.base_center_emp:
    if args.dim_nl==1 and args.dim_internal==1:
        filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_58_13', 'trained_controller.pt')
    elif args.dim_nl==2 and args.dim_internal==4:
        filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_57_38', 'trained_controller.pt')
    elif args.dim_nl==8 and args.dim_internal==8:
        # empirical controller avoids collisions
        # filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_10_10_09_56_16', 'trained_controller.pt')
        # empirical controller does not avoid collisions
        filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_10_11_10_41_10', 'trained_controller.pt')
    res_dict_loaded = torch.load(filename_load)
    mean = np.array([])
    for name in training_param_names:
        mean = np.append(mean, res_dict_loaded[name].cpu().detach().numpy().flatten())
    state_dict = q0.state_dict()
    state_dict['loc'] = torch.Tensor(mean.reshape(1, -1))
    # state_dict['log_scale'] = state_dict['log_scale'] - 100 # TODO
    state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO
    print('stddev', torch.exp(state_dict['log_scale'][0][0:10]))
    print('loaded mean', mean[0:10])
    q0.load_state_dict(state_dict)
else:
    state_dict = q0.state_dict()
    state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO


# set up normflow
nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
nfm.to(device)  # Move model on GPU if available

msg = '\n[INFO] Norm flows setup: num transformations: %i' % args.num_flows
msg += ' -- flow type: ' + args.flow_type if args.num_flows>0 else ' -- flow type: None'
msg += ' -- flow activation: ' + args.flow_activation
msg += ' -- base dist: ' + q0.__class__.__name__ + ' -- base is prior: ' + str(args.base_is_prior)
msg += ' -- base centered at emp: ' + str(args.base_center_emp) + ' -- learn base: ' + str(args.learn_base)
logger.info(msg)

# ------------ 9. Test initial model ------------
# plot closed-loop trajectories by sampling controller from untrained nfm and base distribution
with torch.no_grad():
    for dist, dist_name in zip([q0, nfm], ['base', 'init']):
        logger.info('Plotting closed-loop trajectories for ' + dist_name + ' flow model.')
        if dist_name=='base':
            z = dist.sample(num_samples_nf_eval)
        else:
            z, _ = dist.sample(num_samples_nf_eval)
        z_mean = torch.mean(z, axis=0)
        _, xs_z_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
        )
        _, xs_z_mean_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
        )
        plot_trajectories(
            torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
            dataset.xbar, sys.n_agents, text='CL - ' + dist_name + ' flow', T=t_ext,
            save_folder=save_folder, filename='CL_'+dist_name+'.pdf',
            obstacle_centers=bounded_loss_fn.obstacle_centers,
            obstacle_covs=bounded_loss_fn.obstacle_covs,
            plot_collisions=True, min_dist=bounded_loss_fn.min_dist
        )

# evaluate on the train data
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

# ------------ 10. Train NormFlows ------------
# Train model
anneal_iter = int(args.epochs/2)
annealing = False # NOTE
weight_decay = 0
msg = '[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
msg += ' -- learning rate: %.6f' % args.lr + ' -- weight decay: %.6f' % weight_decay
logger.info(msg)

nf_loss_hist = [None]*args.epochs

optimizer = torch.optim.Adam(nfm.parameters(), lr=args.lr, weight_decay=weight_decay)
with tqdm(range(args.epochs)) as t:
    for it in t:
        optimizer.zero_grad()
        t_now = time.time()
        if annealing:
            nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + it / anneal_iter]))
        else:
            nf_loss = nfm.reverse_kld(num_samples_nf_train)
        # print('reverse KLD time ', time.time()-t_now)
        t_now = time.time()
        nf_loss.backward()
        # print('backward time ', time.time()-t_now)
        # for name, param in nfm.named_parameters():
        #     if not torch.isfinite(param.grad).all():
        #         logger.info('grad for ' + name + 'is infinite.')
        #     if param.isnan().any():
        #         logger.info(param, ' is nan in iter ', it)
        optimizer.step()

        nf_loss_hist[it] = nf_loss.to('cpu').data.numpy()

        # Eval and log
        if (it + 1) % args.log_epoch == 0 or it+1==args.epochs:
            with torch.no_grad():
                # evaluate some sampled controllers
                z, _ = nfm.sample(num_samples_nf_eval)
                loss_z, xs_z = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=train_data,
                    loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
                )
                # evaluate mean of sampled controllers
                z_mean = torch.mean(z, axis=0).reshape(1, -1)
                print(z_mean[0,0:10])
                loss_z_mean, xs_z_mean = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=train_data,
                    loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
                )

            # log nf loss
            elapsed = t.format_dict['elapsed']
            elapsed_str = t.format_interval(elapsed)
            msg = 'Iter %i' % (it+1) + ' --- elapsed time: ' + elapsed_str  + ' --- norm flow loss: %f'  % nf_loss.item()
            msg += ' --- train loss %f' % loss_z + ' --- train loss of mean %f' % loss_z_mean
            logger.info(msg)

            # save nf model
            name = 'final' if it+1==args.epochs else 'itr '+str(it+1)
            if name == 'final':
                torch.save(nfm.state_dict(), os.path.join(save_folder, name+'_nfm'))
            # plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(nf_loss_hist, label='loss')
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'loss.pdf'))
            plt.show()
            # plot closed_loop
            _, xs_z_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
            )
            _, xs_z_mean_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
            )
            plot_trajectories(
                torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
                dataset.xbar, sys.n_agents, text="CL - "+name, T=t_ext,
                save_folder=save_folder, filename='CL_'+name+'.pdf',
                obstacle_centers=bounded_loss_fn.obstacle_centers,
                obstacle_covs=bounded_loss_fn.obstacle_covs,
                plot_collisions=True, min_dist=bounded_loss_fn.min_dist
            )

# ------ 11. evaluate the trained model ------
# evaluate on the train data
logger.info('\n[INFO] evaluating the trained flow on %i training rollouts.' % args.num_rollouts)
train_loss, train_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % train_loss
if args.col_av:
    msg += ' -- total number of collisions = %i' % train_num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained flow on %i test rollouts.' % test_data.shape[0])
test_loss, test_num_col = eval_norm_flow(
    nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=test_data,
    num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=args.col_av
)
msg = 'Average loss: %.4f' % test_loss
if args.col_av:
    msg += ' -- total number of collisions = %i' % test_num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
with torch.no_grad():
    z, _ = nfm.sample(100)
    z_mean = torch.mean(z, axis=0)
    _, xs_z_plot = eval_norm_flow(
        sys=sys, ctl_generic=ctl_generic, data=plot_data,
        loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
    )
    _, xs_z_mean_plot = eval_norm_flow(
        sys=sys, ctl_generic=ctl_generic, data=plot_data,
        loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
    )
    plot_trajectories(
        torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
        dataset.xbar, sys.n_agents, text='CL - trained flow', T=t_ext,
        save_folder=save_folder, filename='CL_trained.pdf',
        obstacle_centers=bounded_loss_fn.obstacle_centers,
        obstacle_covs=bounded_loss_fn.obstacle_covs,
        plot_collisions=True, min_dist=bounded_loss_fn.min_dist
    )