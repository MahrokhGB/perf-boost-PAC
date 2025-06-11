import sys, os, logging, torch, math
import normflows as nf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist.mynf import NormalizingFlow
from inference_algs.normflow_assist import GibbsWrapperNF, eval_norm_flow


def load_nfm(load_path):
    # ----- Load setup -----
    setup_dict = torch.load(os.path.join(load_path, 'setup'))
    trained_nfm = torch.load(os.path.join(load_path, 'trained_nfm'))
    print(trained_nfm.keys())
    

    # ----- SET UP LOGGER -----
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
    logger = WrapLogger(None)

    logger.info('---------- Plotting ----------\n')
    torch.manual_seed(setup_dict['random_seed'])


    # ------------ 1. Basics ------------
    # Dataset
    dataset = RobotsDataset(random_seed=setup_dict['random_seed'], horizon=setup_dict['horizon'], std_ini=setup_dict['std_init_plant'], n_agents=2)
    # divide to train and test
    train_data_full, test_data = dataset.get_data(num_train_samples=setup_dict['num_rollouts'], num_test_samples=500)
    train_data_full, test_data = train_data_full.to(device), test_data.to(device)

    # Plant
    plant_input_init = None     # all zero
    plant_state_init = None     # same as xbar
    sys = RobotsSystem(
        xbar=dataset.xbar, x_init=plant_state_init,
        u_init=plant_input_init, linear_plant=setup_dict['linearize_plant'], k=setup_dict['spring_const']
    ).to(device)

    # Controller
    assert setup_dict['cont_type']=='PerfBoost'
    ctl_generic = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=setup_dict['dim_internal'], dim_nl=setup_dict['dim_nl'],
        initialization_std=setup_dict['cont_init_std'],
        output_amplification=20, train_method='normflow'
    ).to(device)
    num_params = ctl_generic.num_params
    logger.info('[INFO] Controller is of type ' + setup_dict['cont_type'] + ' and has %i parameters.' % num_params)

    # Loss
    Q = torch.kron(torch.eye(setup_dict['n_agents']), torch.eye(4)).to(device)   # TODO: move to args and print info
    x0 = dataset.x0.reshape(1, -1).to(device)
    sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
    sat_bound += 0 if setup_dict['alpha_col'] is None else setup_dict['alpha_col']
    sat_bound += 0 if setup_dict['alpha_obst'] is None else setup_dict['alpha_obst']
    sat_bound = sat_bound/20
    logger.info('Loss saturates at: '+str(sat_bound))
    bounded_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=setup_dict['alpha_u'], xbar=dataset.xbar,
        loss_bound=setup_dict['loss_bound'], sat_bound=sat_bound.to(device),
        alpha_col=setup_dict['alpha_col'], alpha_obst=setup_dict['alpha_obst'],
        min_dist=setup_dict['min_dist'] if setup_dict['col_av'] else None,
        n_agents=sys.n_agents if setup_dict['col_av'] else None,
    )
    C = bounded_loss_fn.loss_bound

    # Prior
    if setup_dict['cont_type'] in ['Affine', 'NN']:
        training_param_names = ['weight', 'bias']
        prior_dict = {
            'type':'Gaussian', 'type_w':'Gaussian',
            'type_b':'Gaussian_biased',
            'weight_loc':0, 'weight_scale':1,
            'bias_loc':0, 'bias_scale':5,
        }
    else:
        if setup_dict['data_dep_prior']:
            if setup_dict['dim_nl']==8 and setup_dict['dim_internal']==8:
                if setup_dict['num_rollouts_prior']==5:
                    filename_load = os.path.join(save_path, 'empirical', 'pretrained', 'trained_controller.pt')
                    res_dict_loaded = torch.load(filename_load)
        if setup_dict['nominal_prior']:
            res_dict_loaded = []
            if setup_dict['dim_nl']==8 and setup_dict['dim_internal']==8:
                for _, dirs, _ in os.walk(os.path.join(save_path, 'nominal')):
                    for dir in dirs:
                        filename_load = os.path.join(save_path, 'nominal', dir, 'trained_controller.pt')
                        res_dict_loaded.append(torch.load(filename_load))
            logger.info('[INFO] Loaded '+str(len(res_dict_loaded))+' nominal controllers.')
        prior_dict = {'type':'Gaussian'}
        training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        for name in training_param_names:
            if setup_dict['data_dep_prior']:
                prior_dict[name+'_loc'] = res_dict_loaded[name]
                prior_dict[name+'_scale'] = setup_dict['prior_std']
            elif setup_dict['nominal_prior']:
                logger.info(
                    '[INFO] Prior distribution is the distribution over nominal controllers, with std scaled by %.4f.' % setup_dict['nominal_prior_std_scale']
                )
                vals = torch.stack([res[name] for res in res_dict_loaded], dim=0)
                # val and std computed elementwise. same shape as the training param
                prior_dict[name+'_loc'] = vals.mean(dim=0)  
                prior_dict[name+'_scale'] = vals.std(dim=0, correction=1) * setup_dict['nominal_prior_std_scale']
            else:
                prior_dict[name+'_loc'] = 0
                prior_dict[name+'_scale'] = setup_dict['prior_std']

    # Posterior
    gibbs_posteior = GibbsPosterior(
        loss_fn=bounded_loss_fn,
        lambda_=setup_dict['gibbs_lambda'],
        prior_dict=prior_dict,
        # attributes of the CL system
        controller=ctl_generic, sys=sys,
        # misc
        logger=logger,
    )

    # Wrap Gibbs distribution to be used in normflows
    target = None
    # GibbsWrapperNF(
    #     target_dist=gibbs_posteior, train_dataloader=train_dataloader,
    #     prop_scale=trained_nfm['prop_scale'], prop_shift=trained_nfm['prop_shift']
    # )

    # ------------ load NormFlows ------------
    flows = []
    for flow_num in range(setup_dict['num_flows']):
        if setup_dict['flow_type'] == 'Radial':
            # flows += [nf.flows.Radial((num_params,), act=setup_dict['flow_activation'])]
            raise NotImplementedError
        elif setup_dict['flow_type'] == 'Planar': # f(z) = z + u * h(w * z + b)
            '''
            Default values:
                - u: uniform(-sqrt(2), sqrt(2))
                - w: uniform(-sqrt(2/num_prams), sqrt(2/num_prams))
                - b: 0
                - h: setup_dict['flow_activation (tanh or leaky_relu)
            '''
            flow = nf.flows.Planar((num_params,), u=setup_dict['planar_flow_scale']*(2*torch.rand(num_params)-1), act=setup_dict['flow_activation'])
            for param_name in ['u', 'w', 'b']:
                setattr(
                    flow, 
                    param_name, 
                    torch.nn.Parameter(
                        trained_nfm['flows.'+str(flow_num)+'.'+param_name]
                    )
                )
            flows += [flow]
        elif setup_dict['flow_type'] == 'NVP':
            # # Neural network with two hidden layers having 64 units each
            # # Last layer is initialized by zeros making training more stable
            # param_map = nf.nets.MLP([math.ceil(num_params/2), 64, 64, num_params], init_zeros=True, act=setup_dict['flow_activation'])
            # # Add flow layer
            # flows.append(nf.flows.AffineCouplingBlock(param_map))
            # # Swap dimensions
            # flows.append(nf.flows.Permute(2, mode='swap'))
            raise NotImplementedError
        else:
            raise NotImplementedError

    # base distribution
    q0 = nf.distributions.DiagGaussian(num_params, trainable=False)
    q0.loc = trained_nfm['q0.loc']
    q0.log_scale = trained_nfm['q0.log_scale']

    # set up normflow
    nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
    nfm.to(device)  # Move model on GPU if available

    return nfm, sys, ctl_generic, bounded_loss_fn, train_data_full, test_data



# num_samples = np.logspace(5, 10, num=6, base=2)
# ub = [0.65, 0.62, 0.60, 0.58, 0.57, 0.56]

# fig, axs = plt.subplots(figsize=(5,4))
# plt.scatter(num_samples, ub)
# plt.xlabel('Number of samples')
# plt.ylabel('Upper bound')
# plt.title('Upper bound vs number of samples for delta = 0.1')
# plt.savefig('foo.png')



# ------ format ------
plt.rcParams['text.usetex'] = True
sns.set_theme(
    context='paper', style='whitegrid', palette='bright', 
    font='sans-serif', font_scale=1.4, color_codes=True, rc=None, 
)
sns.set_style({'grid.linestyle': '--'})
mpl.rc('font', family='serif', serif='Times New Roman')


# ------ init ------
Ss = [32]
num_sampled_controllers = 100
ub_dict = dict.fromkeys([
    'number of training rollouts', 'prior_type', 'bounded true loss', 
    # 'av_test_loss_original', 
    'ub', 
    # 'ub_sb', 
    'sampled controller number', 
    'delta', 'setup', 
    # 'emp_loss_bounded', 'emp_loss_original'
])
# epsilons = [0.1, 0.2]
# ptypes = ['Uniform', 'Gaussian_biased_wide']
# num_rows = len(epsilons)*len(Ss)*len(ptypes)*num_sampled_controllers
num_rows = num_sampled_controllers
for key in ub_dict.keys():
    ub_dict[key] = [None]*num_rows
ind = 0


# ------ loop over setups ------
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
filenames = ['PerfBoost_06_02_10_03_08']

for filename in filenames:
    print('filename', filename)
    load_path = save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'normflow', filename)
    nfm, sys, ctl_generic, bounded_loss_fn, train_data_full, test_data = load_nfm(load_path)
    z, _ = nfm.sample(num_sampled_controllers)
    z_mean = torch.mean(z, axis=0)
    loss_val, num_col = eval_norm_flow(
        sys=sys, ctl_generic=ctl_generic, data=test_data,
        loss_fn=bounded_loss_fn, count_collisions=True, 
        return_traj=False, params=z, return_av=False
    )

    # set properties
    for sample_num in range(num_sampled_controllers):
        ub_dict['delta'] = 0.1
        ub_dict['number of training rollouts'][ind] = 32
        ub_dict['prior_type'][ind] = 'Gaussian'
        ub_dict['bounded true loss'][ind] = loss_val[sample_num].item()
        # ub_dict['av_test_loss_original'][ind] = res['av_test_loss_original'][sample_num]
        ub_dict['ub'][ind] = 0.3664 #0.5848
        # ub_dict['ub_sb'][ind] = ubsb
        ub_dict['sampled controller number'][ind] = sample_num
        # ub_dict['emp_loss_bounded'][ind] = emp_loss_bounded[s_tmp]
        # ub_dict['emp_loss_original'][ind] = emp_loss_original[s_tmp]
        ub_dict['setup'][ind] = 'Prior ' + ub_dict['prior_type'][ind] + ', $\delta$ = ' + str(ub_dict['delta'])
        ind = ind + 1
    print(ub_dict['bounded true loss'])
# assert ind==num_rows
df = pd.DataFrame(ub_dict)


# ------------------
# ------ PLOT ------
# ------------------
# # mark sampled controllers performance
# g = sns.stripplot(
#     data=df, x='number of training rollouts', y='bounded true loss',
#     # hue='setup', palette='hls', alpha=0.9, dodge=True
# )

# only used to create a proper FacetGrid
g = sns.catplot(
    data=df, x='number of training rollouts', y='ub', #col='disturbance type',
    hue='setup', kind='box', height=4, aspect=1.5,
    sharey=False, palette='hls', legend=False
)

# mark upper bounds
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='ub',
    linewidth=2, linecolor='k', alpha = 0.6,
    hue='setup', legend=False, # dodge=True, 
)


# mark sampled controllers performance
g.map_dataframe(
    sns.stripplot, x='number of training rollouts', y='bounded true loss',
    hue='setup', palette='hls', alpha=0.9, dodge=True
)

# # add legend for the upper bound
# custom_line = [Line2D([0], [0], color='k', lw=2)]

# # ------ legends and titles ------
# ax = g.axes[0,0]
# # add vertical lines between groups
# [ax.axvline(x+.5,color='k', alpha=0.2) for x in ax.get_xticks()]

# # change xtick labels to integer without the leading 0 
# labels = [item.get_text() for item in ax.get_xticklabels()]
# ax.set_xticks(ax.get_xticks(), [str(int(float(label))) for label in labels])

# # axis labels
# ax.set_xlabel(r'Number of training sequences ($s$)')
# ax.set_ylabel(r'True cost ($\mathcal{L}$)')

# # legend
# handles, labels = ax.get_legend_handles_labels()
# handles = handles+custom_line
# labels = labels + ['Upper bound']
# l = plt.legend(
#     handles, labels, bbox_to_anchor=(0.62, 0.98), 
#     loc=2, borderaxespad=0.
# )
# ---------------------------------
save_folder = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'plots')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
filename = os.path.join(save_folder, 'ub.pdf')
plt.savefig(filename)
# plt.show()
plt.close()
print('plotting completed')