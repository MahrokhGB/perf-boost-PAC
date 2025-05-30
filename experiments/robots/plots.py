# import sys, os, logging, torch, math
# from datetime import datetime
# from torch.utils.data import DataLoader
# import normflows as nf

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(1, BASE_DIR)

# from config import device
# from inference_algs.normflow_assist import fit_norm_flow
# from arg_parser import argument_parser, print_args
# from plants import RobotsSystem, RobotsDataset
# from utils.plot_functions import *
# from controllers import PerfBoostController, AffineController, NNController
# from loss_functions import RobotsLossMultiBatch
# from utils.assistive_functions import WrapLogger
# from inference_algs.distributions import GibbsPosterior
# from ub_ub.ub_utils import get_max_lambda
# from inference_algs.normflow_assist.mynf import NormalizingFlow
# from inference_algs.normflow_assist import GibbsWrapperNF


# save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')

# # ----- Load setup -----
# load_path = save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'normflow', 'PerfBoost_01_26_14_41_08')
# setup_dict = torch.save(os.path.join(load_path, 'setup'))

# # ----- SET UP LOGGER -----
# now = datetime.now().strftime("%m_%d_%H_%M_%S")
# save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
# save_folder = os.path.join(save_path, 'plots'+now)
# os.makedirs(save_folder)
# logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
# logger = logging.getLogger('ren_controller_')
# logger.setLevel(logging.DEBUG)
# logger = WrapLogger(logger)

# logger.info('---------- Plotting ----------\n')
# torch.manual_seed(setup_dict['random_seed'])


# # ------------ 1. Basics ------------
# # Dataset
# dataset = RobotsDataset(random_seed=setup_dict['random_seed'], horizon=setup_dict['horizon'], std_ini=setup_dict['std_init_plant'], n_agents=2)
# # divide to train and test
# train_data_full, test_data = dataset.get_data(num_train_samples=setup_dict['num_rollouts'], num_test_samples=500)
# train_data_full, test_data = train_data_full.to(device), test_data.to(device)
# # data for plots
# t_ext = setup_dict['horizon'] * 4
# plot_data = torch.zeros(1, t_ext, train_data_full.shape[-1], device=device)
# plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
# plot_data = plot_data.to(device)

# # Plant
# plant_input_init = None     # all zero
# plant_state_init = None     # same as xbar
# sys = RobotsSystem(
#     xbar=dataset.xbar, x_init=plant_state_init,
#     u_init=plant_input_init, linear_plant=setup_dict['linearize_plant'], k=setup_dict['spring_const']
# ).to(device)

# # Controller
# assert setup_dict['cont_type']=='PerfBoost'
# ctl_generic = PerfBoostController(
#     noiseless_forward=sys.noiseless_forward,
#     input_init=sys.x_init, output_init=sys.u_init,
#     dim_internal=setup_dict['dim_internal'], dim_nl=setup_dict['dim_nl'],
#     initialization_std=setup_dict['cont_init_std'],
#     output_amplification=20, train_method='normflow'
# ).to(device)
# num_params = ctl_generic.num_params
# logger.info('[INFO] Controller is of type ' + setup_dict['cont_type'] + ' and has %i parameters.' % num_params)

# # Loss
# Q = torch.kron(torch.eye(setup_dict['n_agents']), torch.eye(4)).to(device)   # TODO: move to args and print info
# x0 = dataset.x0.reshape(1, -1).to(device)
# sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
# sat_bound += 0 if setup_dict['alpha_col'] is None else setup_dict['alpha_col']
# sat_bound += 0 if setup_dict['alpha_obst'] is None else setup_dict['alpha_obst']
# sat_bound = sat_bound/20
# logger.info('Loss saturates at: '+str(sat_bound))
# bounded_loss_fn = RobotsLossMultiBatch(
#     Q=Q, alpha_u=setup_dict['alpha_u'], xbar=dataset.xbar,
#     loss_bound=setup_dict['loss_bound'], sat_bound=sat_bound.to(device),
#     alpha_col=setup_dict['alpha_col'], alpha_obst=setup_dict['alpha_obst'],
#     min_dist=setup_dict['min_dist'] if setup_dict['col_av'] else None,
#     n_agents=sys.n_agents if setup_dict['col_av'] else None,
# )
# C = bounded_loss_fn.loss_bound



# # ------------ load NormFlows ------------
# num_samples_nf_train = 100
# num_samples_nf_eval = num_samples_nf_train 

# flows = []
# for i in range(setup_dict['num_flows']):
#     if setup_dict['flow_type'] == 'Radial':
#         flows += [nf.flows.Radial((num_params,), act=setup_dict['flow_activation'])]
#     elif setup_dict['flow_type'] == 'Planar': # f(z) = z + u * h(w * z + b)
#         '''
#         Default values:
#             - u: uniform(-sqrt(2), sqrt(2))
#             - w: uniform(-sqrt(2/num_prams), sqrt(2/num_prams))
#             - b: 0
#             - h: setup_dict['flow_activation (tanh or leaky_relu)
#         '''
#         flows += [nf.flows.Planar((num_params,), u=U_SCALE*(2*torch.rand(num_params)-1), act=setup_dict['flow_activation'])]
#     elif setup_dict['flow_type'] == 'NVP':
#         # Neural network with two hidden layers having 64 units each
#         # Last layer is initialized by zeros making training more stable
#         param_map = nf.nets.MLP([math.ceil(num_params/2), 64, 64, num_params], init_zeros=True, act=setup_dict['flow_activation'])
#         # Add flow layer
#         flows.append(nf.flows.AffineCouplingBlock(param_map))
#         # Swap dimensions
#         flows.append(nf.flows.Permute(2, mode='swap'))
#     else:
#         raise NotImplementedError

# # base distribution
# q0 = nf.distributions.DiagGaussian(num_params, trainable=setup_dict['learn_base'])
# # base distribution same as the prior
# if setup_dict['base_is_prior']:
#     BASE_STDP_SCALE = 1/PRIOR_STDP_SCALE/100 # TODO
#     logger.info('[INFO] Base distribution is similar to the prior, with std scaled by %.4f.' % BASE_STDP_SCALE)
#     state_dict = q0.state_dict()
#     state_dict['loc'] = gibbs_posteior.prior.mean().reshape(1, -1)
#     state_dict['log_scale'] = torch.log(gibbs_posteior.prior.stddev().reshape(1, -1)*BASE_STDP_SCALE) 
#     q0.load_state_dict(state_dict)
# # base distribution centered at the empirical or nominal controller
# elif setup_dict['base_center_emp'] or setup_dict['base_center_nominal']:
#     # get filename to load
#     if setup_dict['base_center_emp']:
#         if setup_dict['dim_nl']==1 and setup_dict['dim_internal']==1:
#             filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_58_13', 'trained_controller.pt')
#         elif setup_dict['dim_nl']==2 and setup_dict['dim_internal']==4:
#             filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_08_29_14_57_38', 'trained_controller.pt')
#         elif setup_dict['dim_nl']==8 and setup_dict['dim_internal']==8:
#             # # empirical controller avoids collisions
#             # filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_01_14_16_26_11', 'trained_controller.pt')
#             # empirical controller does not avoid collisions
#             filename_load = os.path.join(save_path, 'empirical', 'PerfBoost_01_19_11_31_25', 'trained_controller.pt')
#     if setup_dict['base_center_nominal']:
#         if setup_dict['dim_nl']==8 and setup_dict['dim_internal']==8:
#             filename_load = os.path.join(save_path, 'nominal', 'PerfBoost_01_22_15_25_52', 'trained_controller.pt')
#     # load the controller
#     res_dict_loaded = torch.load(filename_load)
#     # set the mean of the base distribution to the controller
#     mean = np.array([])
#     for name in training_param_names:
#         mean = np.append(mean, res_dict_loaded[name].cpu().detach().numpy().flatten())
#     state_dict = q0.state_dict()
#     state_dict['loc'] = torch.Tensor(mean.reshape(1, -1))
#     # set the scale of the base distribution
#     # state_dict['log_scale'] = state_dict['log_scale'] - 100 # TODO
#     state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO
#     q0.load_state_dict(state_dict)
# else:
#     state_dict = q0.state_dict()
#     state_dict['log_scale'] = torch.log(torch.abs(state_dict['loc'])*STD_SCALE) # TODO

# # set up normflow
# nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
# nfm.to(device)  # Move model on GPU if available

import matplotlib.pyplot as plt
import numpy as np

num_samples = np.logspace(5, 10, num=6, base=2)
ub = [0.65, 0.62, 0.60, 0.58, 0.57, 0.56]

fig, axs = plt.subplots(figsize=(5,4))
plt.scatter(num_samples, ub)
plt.xlabel('Number of samples')
plt.ylabel('Upper bound')
plt.title('Upper bound vs number of samples for delta = 0.1')
plt.savefig('foo.png')