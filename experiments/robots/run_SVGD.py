import sys, os, logging, torch, math
from datetime import datetime
from torch.utils.data import DataLoader
from pyro.distributions import Normal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from utils.plot_functions import *
from plants import RobotsSystem, RobotsDataset
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from arg_parser import argument_parser, print_args
from inference_algs.distributions import GibbsPosterior
from controllers import PerfBoostController, AffineController, NNController, SVGDCont

"""

"""

TRAIN_METHOD = 'SVGD'

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
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # TODO
# train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False)

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
gibbs_lambda = gibbs_lambda_star
logger.info('gibbs_lambda: %.2f' % gibbs_lambda + ' (use lambda_*)' if gibbs_lambda == gibbs_lambda_star else '')
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda, prior_dict=prior_dict,
    # attributes of the CL system
    controller=ctl_generic, sys=sys,
    # misc
    logger=logger,
)


# ****** INIT SVGD ******
num_particles = 1
# lr = 1e-2
return_best = True
# initialize trainable params
initialization_std = 0.1 if args.obst_av else 1.0
dim = (num_particles, ctl_generic.num_params)

initial_particles = Normal(0, initialization_std).sample(dim).to(device)

svgd_cont = SVGDCont(
    gibbs_posteior=gibbs_posteior,
    num_particles=num_particles, logger=logger,
    optimizer='Adam', lr=args.lr, lr_decay=None, #TODO: add decay
    initial_particles=initial_particles, kernel='RBF', bandwidth=None,
)
msg = '\n[INFO] SVGD: delta: %.2f' % args.delta + ' -- num particles: %2.f' % num_particles
msg += ' -- initialization std: %.4f' % initialization_std


# ****** TRAIN SVGD ******

logger.info('------------ Begin training ------------')
svgd_cont.fit(
    dataloader=train_dataloader,
    over_fit_margin=None, cont_fit_margin=None, max_iter_fit=None,
    return_best=return_best, log_period=args.log_epoch, epochs=args.epochs,
    valid_data=train_data   # NOTE: validate model using the entire train data
)
logger.info('Training completed.')

# # ------ Save trained model ------
# particles = svgd_cont.particles.detach().clone()
# res_dict = {
#     'particles':particles,
#     'num_rollouts':num_rollouts,
#     'Q':Q, 'alpha_u':alpha_u,
#     'alpha_ca':alpha_ca, 'alpha_obst':alpha_obst,
#     'n_xi':n_xi, 'l':l, 'initialization_std':initialization_std
# }
# # save file name
# if fname is not None:
#     filename_save = fname+'_'+str(num_particles)+'particles.pt'
# else:
#     filename_save = exp_name+'_SVGD_'+str(num_particles)+'particles_T'+str(t_end)+'_S'+str(num_rollouts)
#     filename_save += '_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pt'
# file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models')
# path_exist = os.path.exists(file_path)
# if not path_exist:
#     os.makedirs(file_path)
# filename_save = os.path.join(file_path, filename_save)
# torch.save(res_dict, filename_save)
# logger.info('model saved.')

# eval on train data
bounded_train_loss = svgd_cont.eval_rollouts(train_data)
original_train_loss = svgd_cont.eval_rollouts(train_data, loss_fn=original_loss_fn)
logger.info('Final results on the entire train data: Bounded train loss = {:.4f}, original train loss = {:.4f}'.format(
    bounded_train_loss, original_train_loss
))

# ------------ 5. Test Dataset ------------

# bounded_test_loss = svgd_cont.eval_rollouts(test_data)
# original_test_loss = svgd_cont.eval_rollouts(test_data, loss_fn=original_loss_fn)
# msg = 'True bounded test loss = {:.4f}, '.format(bounded_test_loss)
# msg += 'true original test loss = {:.2f} '.format(original_test_loss)
# msg += '(approximated using {:3.0f} test rollouts).'.format(test_data.shape[0])
# logger.info(msg)