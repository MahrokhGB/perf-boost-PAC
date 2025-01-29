import sys, os, torch, math
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from inference_algs.distributions import GibbsPosterior
from inference_algs.normflow_assist.mynf import NormalizingFlow
import normflows as nf
from inference_algs.normflow_assist import GibbsWrapperNF
from filename_lookup import get_filename
from ub_utils import get_mcdim_ub


delta = 0.01
ACT = 'tanh'
LEARN_BASE = True
prior_std = 3

TRAIN_METHOD = 'normflow'
cont_type = 'PerfBoost'
# S = np.logspace(start=3, stop=9, num=7, dtype=int, base=2)
S = [256, 512]
# S = np.logspace(start=3, stop=8, num=6, dtype=int, base=2)

now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(BASE_DIR, 'experiments', 'robots', 'ub_ub', 'saved_results')

num_samples_nf_eval = 100
results = dict.fromkeys(['delta', 'number of training rollouts', 'test loss', 'test num collisions', 'ub on ub', 'ub uniform', 'ub unnormpost', 'ub const', '-1/lambda ln(Zhat)'])
for res_key in results.keys():
    results[res_key] = [None]*len(S)*num_samples_nf_eval
row_num = 0

mcdim_ubs = []
for num_rollouts in S:
    FILE_NAME = get_filename(
        delta=delta,
        learn_base=LEARN_BASE, act=ACT, num_rollouts=num_rollouts, prior_std=prior_std)
    assert not FILE_NAME is None, num_rollouts

    # ----- Load -----
    # load training setup
    setup_loaded = torch.load(
        os.path.join(save_path, 'normflow', FILE_NAME, 'setup'),
        map_location=torch.device('cpu')
    )
    if not 'flow_activation' in setup_loaded.keys():
        print('[WARN] replace activation by default tanh.')
        setup_loaded['flow_activation'] = 'tanh'
    assert num_rollouts==setup_loaded['num_rollouts']
    assert delta==setup_loaded['delta']
    assert ACT==setup_loaded['flow_activation']

    # load trained nfm
    nfm_loaded = torch.load(
        os.path.join(save_path, 'normflow', FILE_NAME, 'final_nfm'),
        map_location=torch.device('cpu')
    )
    nfm_keys = nfm_loaded.keys()
    print('[INFO] NFM model loaded')
    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(
        random_seed=setup_loaded['random_seed'], horizon=setup_loaded['horizon'],
        std_ini=setup_loaded['std_init_plant'], n_agents=2
    )
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=setup_loaded['num_rollouts'], num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)
    train_dataloader = DataLoader(train_data, batch_size=min(setup_loaded['num_rollouts'], 512), shuffle=False)

    # ------------ 2. Plant ------------
    plant_input_init = None     # all zero
    plant_state_init = None     # same as xbar
    sys = RobotsSystem(
        xbar=dataset.xbar, x_init=plant_state_init,
        u_init=plant_input_init, linear_plant=setup_loaded['linearize_plant'], k=setup_loaded['spring_const']
    ).to(device)

    # ------------ 3. Controller ------------
    if setup_loaded['cont_type']=='PerfBoost':
        ctl_generic = PerfBoostController(
            noiseless_forward=sys.noiseless_forward,
            input_init=sys.x_init, output_init=sys.u_init,
            dim_internal=setup_loaded['dim_internal'], dim_nl=setup_loaded['dim_nl'],
            initialization_std=setup_loaded['cont_init_std'],
            output_amplification=20, train_method=TRAIN_METHOD
        ).to(device)
    elif setup_loaded['cont_type']=='Affine':
        ctl_generic = AffineController(
            weight=torch.zeros(sys.in_dim, sys.state_dim, device=device, dtype=torch.float32),
            bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32),
            train_method=TRAIN_METHOD
        )
    elif setup_loaded['cont_type']=='NN':
        ctl_generic = NNController(
            in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=setup_loaded['layer_sizes'],
            train_method=TRAIN_METHOD
        )
    else:
        raise KeyError('[Err] cont_type must be PerfBoost, NN, or Affine.')

    num_params = ctl_generic.num_params

    # ------------ 4. Loss ------------
    Q = torch.kron(torch.eye(setup_loaded['n_agents']), torch.eye(4)).to(device)
    x0 = dataset.x0.reshape(1, -1).to(device)
    bounded_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=setup_loaded['alpha_u'], xbar=dataset.xbar,
        loss_bound=setup_loaded['loss_bound'], sat_bound=setup_loaded['sat_bound'].to(device),
        alpha_col=setup_loaded['alpha_col'], alpha_obst=setup_loaded['alpha_obst'],
        min_dist=setup_loaded['min_dist'] if setup_loaded['col_av'] else None,
        n_agents=sys.n_agents if setup_loaded['col_av'] else None,
    )
    original_loss_fn = RobotsLossMultiBatch(
        Q=Q, alpha_u=setup_loaded['alpha_u'], xbar=dataset.xbar,
        loss_bound=None, sat_bound=None,
        alpha_col=setup_loaded['alpha_col'], alpha_obst=setup_loaded['alpha_obst'],
        min_dist=setup_loaded['min_dist'] if setup_loaded['col_av'] else None,
        n_agents=sys.n_agents if setup_loaded['col_av'] else None,
    )

    # ------------ 5. Prior ------------
    prior_dict = setup_loaded['prior_dict']

    # ------------ 6. Posterior ------------
    # define target distribution
    gibbs_posteior = GibbsPosterior(
        loss_fn=bounded_loss_fn, lambda_=setup_loaded['gibbs_lambda'], prior_dict=prior_dict,
        # attributes of the CL system
        controller=ctl_generic, sys=sys,
        # misc
        logger=None,
    )

    # Wrap Gibbs distribution to be used in normflows
    target = GibbsWrapperNF(
        target_dist=gibbs_posteior, train_dataloader=train_dataloader,
        prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
    )

    # ------------ 7. Load NormFlow ------------
    # base distribution
    if 'q0.loc' in nfm_keys and 'q0.log_scale' in nfm_keys:
        assert (1, num_params)==nfm_loaded['q0.loc'].shape
        assert (1, num_params)==nfm_loaded['q0.log_scale'].shape
        q0 = nf.distributions.DiagGaussian(num_params)
    else:
        raise NotImplementedError

    # flows
    last_flow = next((dict_key for dict_key in reversed(nfm_keys) if dict_key.startswith('flow')), None)
    num_flows = 0 if last_flow is None else int(last_flow.split('.')[1])+1
    assert num_flows==setup_loaded['num_flows']
    flows = []
    for flow_num in range(num_flows):
        # planar flow
        if set(['flows.'+str(flow_num)+'.u', 'flows.'+str(flow_num)+'.w', 'flows.'+str(flow_num)+'.b']) <= set(nfm_keys):
            flows += [nf.flows.Planar((num_params,), act=setup_loaded['flow_activation'])]
            for flow_key in ['u', 'w', 'b']:
                nfm_loaded['flows.'+str(flow_num)+'.'+flow_key] = nfm_loaded['flows.'+str(flow_num)+'.'+flow_key].reshape(
                    getattr(flows[flow_num], flow_key).shape
                )
        else:
            raise NotImplementedError

    # init dummy NF model
    nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
    # load state dict
    nfm.load_state_dict(nfm_loaded)
    nfm.to(device)  # Move model on GPU if available

    # ------------ 8. Test Results ------------
    print('\n[INFO] evaluating the trained flow on %i test rollouts.' % test_data.shape[0])
    test_loss, test_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=test_data,
        num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn,
        count_collisions=setup_loaded['col_av'], return_av=False
    )
    print('test_loss', sum(test_loss).item()/num_samples_nf_eval)

    # ------------ 9. UB on UB ------------
    lambda_ = setup_loaded['gibbs_lambda']

    # ------------ Mc Diarmid ------------
    mcdim_ub = get_mcdim_ub(
        sys=sys, ctl_generic=ctl_generic, train_data=train_data, bounded_loss_fn=bounded_loss_fn,
        prior=gibbs_posteior.prior, delta=delta, lambda_=lambda_, C=setup_loaded['loss_bound'],
        num_prior_samples=10000, deltahat=delta
    )



#     # ------------ WRONG ub on ub ------------
#     C = setup_loaded['loss_bound']
#     ub_const = 1/lambda_*math.log(1/delta) + lambda_*C**2/8/num_rollouts
#     train_loss, _ = eval_norm_flow(
#         nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
#         num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn,
#         count_collisions=setup_loaded['col_av'], return_av=False
#     )
#     gamma = setup_loaded['gibbs_lambda']/8/num_rollouts + 2/setup_loaded['gibbs_lambda']*math.log(1/delta)
#     ub_on_ub_wrong = train_loss + gamma

#     # ------------ CDC approx ub ------------
#     # DOESN'T WORK. NEEDS TOO MANY SAMPLES
#     # deltahat = 0.1
#     # n_p_min = math.ceil((math.exp(lambda_*C)-1)**2 * math.log(1/deltahat)/2)
#     # ctl_generic.reset()
#     # print(n_p_min)
#     # num_prior_samples = n_p_min
#     # prior = gibbs_posteior.prior
#     # prior_samples = prior.sample(torch.Size([num_prior_samples]))
#     # train_loss_prior_samples, _ = eval_norm_flow(
#     #     sys=sys, ctl_generic=ctl_generic, data=train_data,
#     #     num_samples=None, params=prior_samples, nfm=None,
#     #     loss_fn=bounded_loss_fn,
#     #     count_collisions=setup_loaded['col_av'], return_av=False
#     # )
#     # exp_neg_loss = torch.exp(- setup_loaded['gibbs_lambda']*train_loss_prior_samples)
#     # cdc_ub = ub_const - \
#     #     1/lambda_*math.log(sum(exp_neg_loss) + (math.exp(-lambda_*C)-1)*(math.log(1/deltahat)/2/num_prior_samples)**0.5)

    # # ------------ swapping denom ------------
    # # ctl_generic.emme.hard_reset()

    # num_samples = 100
    # sample_from = 'nf_post' #'prior' # 'nf_post'

    # msg = 'Approximating the partition function (Z) using {:.0f} samples '.format(
    #     num_samples
    # )
    # msg += 'using '+sample_from+' sampling method.'
    # print(msg)

    # # sample controllers
    # if sample_from=='uniform':
    #     raise NotImplementedError
    # elif sample_from=='nf_post':
    #     sampled_controllers, _ = nfm.sample(num_samples)
    # elif sample_from=='q0':
    #     sampled_controllers = nfm.q0.sample(num_samples)
    # elif sample_from=='prior':
    #     sampled_controllers = gibbs_posteior.prior.sample(torch.Size([num_samples]))
    # else:
    #     raise NotImplementedError
    # if len(sampled_controllers.shape)==1:
    #     sampled_controllers = sampled_controllers.reshape(1, sampled_controllers.shape[0])
    # # Zhat
    # loss_sampled_controllers, _ = eval_norm_flow(
    #     sys=sys, ctl_generic=ctl_generic, data=train_data,
    #     num_samples=None, params=sampled_controllers, nfm=None,
    #     loss_fn=bounded_loss_fn,
    #     count_collisions=setup_loaded['col_av'], return_av=False
    # )
    # print('loss_sampled_controllers', sum(loss_sampled_controllers)/num_samples)
    # log_prob = nfm.log_prob(sampled_controllers)

    # log_Zhat_terms = -lambda_ * loss_sampled_controllers - log_prob
    # assert log_Zhat_terms.shape==(num_samples,)
    # log_Zhat_terms_mean = sum(log_Zhat_terms)/num_samples
    # log_Zhat_terms_normalized = log_Zhat_terms - log_Zhat_terms_mean
    # Zhat_terms_normalized = torch.exp(log_Zhat_terms_normalized)
    # Zhat_terms_normalized
    # # 1. uniform weighting
    # log_Zhat_uniform = torch.log(sum(Zhat_terms_normalized)/num_samples) + log_Zhat_terms_mean
    # print('log_Zhat_uniform/lambda', log_Zhat_uniform/lambda_)
    # ub_uniform = ub_const - 1/lambda_*log_Zhat_uniform
    # print('Weighting uniformly: ', ub_uniform)

    # # 2. weight by the unnormalized posterior
    # weights = Zhat_num/sum(Zhat_num)
    # log_Zhat_unnormpost = torch.log(sum(Zhat_terms*weights))-log_Zhat_den2
    # ub_unnormpost = ub_const - 1/lambda_*log_Zhat_unnormpost
    # print(
    #     'Weighting according to unnormalized posterior: ',
    #     ub_unnormpost
    # )

    # # 3. weighting by variational posterior
    # weights = Zhat_den/sum(Zhat_den)
    # weights = weights/sum(weights)
    # print(
    #     'Weighting according to variational posterior: ',
    #     np.sum(approx_Z_theta*weights)
    # )



    for i in range(num_samples_nf_eval):
        results['delta'][row_num] = delta
        results['number of training rollouts'][row_num] = num_rollouts
        results['test loss'][row_num] = test_loss[i].item()
        results['test num collisions'][row_num] = test_num_col[i]
        results['ub on ub'][row_num] = 0.573 if num_rollouts==512 else 0.582
        # results['ub on ub'][row_num] = mcdim_ub.item()
        # results['ub const'][row_num] = ub_const.item()
        # results['-1/lambda ln(Zhat)'] = (- 1/lambda_*math.log(Z_hat_norm) + mean_loss).item()
        # results['ub on ub wrong'][row_num] = ub_on_ub_wrong[i].to(torch.device('cpu'))
        # results['ub uniform'][row_num] = ub_uniform.item()
        # results['ub unnormpost'][row_num] = ub_unnormpost.item()
        row_num += 1
# ------------------
# ------ PLOT ------
# ------------------
df = pd.DataFrame(results)
# only used to create a proper FacetGrid
g = sns.catplot(
    data=df, x='number of training rollouts', y='ub on ub', #col='delta',
    hue='delta', kind='box', height=4, aspect=1.5,
    sharey=False, palette='hls', legend=False
)

# mark upper bounds
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='ub on ub',
    hue='delta', linewidth=2, linecolor='blue', alpha = 0.6,
    legend=False, # dodge=True,
)
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='ub const',
    hue='delta', linewidth=2, linecolor='k', alpha = 0.6,
    legend=False, # dodge=True,
)
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='-1/lambda ln(Zhat)',
    hue='delta', linewidth=2, linecolor='cyan', alpha = 0.6,
    legend=False, # dodge=True,
)
# g.map_dataframe(
#     sns.boxenplot, x='number of training rollouts', y='ub unnormpost',
#     hue='delta', linewidth=2, linecolor='b', alpha = 0.6,
#     legend=False, # dodge=True,
# )

# mark sampled controllers performance
g.map_dataframe(
    sns.stripplot, x='number of training rollouts', y='test loss',
    hue='delta', palette='hls', alpha=0.9, dodge=True
)

# add legend for the upper bound
custom_line_blue = [Line2D([0], [0], color='blue', lw=2)]
custom_line_black = [Line2D([0], [0], color='k', lw=2)]
custom_line_cyan = [Line2D([0], [0], color='cyan', lw=2)]

# ------ legends and titles ------
ax = g.axes[0,0]
# add vertical lines between groups
[ax.axvline(x+.5,color='k', alpha=0.2) for x in ax.get_xticks()]

# # change xtick labels to integer without the leading 0
# labels = [item.get_text() for item in ax.get_xticklabels()]
# ax.set_xticks(ax.get_xticks(), [str(int(float(label))) for label in labels])

# axis labels
ax.set_xlabel(r'Number of training sequences ($s$)')
ax.set_ylabel(r'True cost ($\mathcal{L}$)')

# legend
handles, labels = ax.get_legend_handles_labels()
handles = handles+custom_line_blue+custom_line_black+custom_line_cyan
labels = labels + ['Upper bound', 'UB constant', '-1/lambda ln(Zhat)']
l = plt.legend(
    handles, labels, bbox_to_anchor=(0.62, 0.98),
    loc=2, borderaxespad=0.
)
# ---------------------------------
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
filename = os.path.join(save_path, 'ub.pdf')
plt.savefig(filename)
plt.show()
