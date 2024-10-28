import sys, os, torch, math
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from inference_algs.normflow_assist.myplanar import MyPlanar
import normflows as nf
from inference_algs.normflow_assist import GibbsWrapperNF

ACT = 'tanh' # leaky_relu

delta = 0.01
TRAIN_METHOD = 'normflow'
cont_type = 'PerfBoost'
S = np.logspace(start=3, stop=9, num=7, dtype=int, base=2)
# S = [8, 16]

num_samples_nf_eval = 100
results = dict.fromkeys(['delta', 'number of training rollouts', 'test loss', 'test num collisions', 'ub on ub'])
for res_key in results.keys():
    results[res_key] = [None]*len(S)*num_samples_nf_eval
row_num = 0

for num_rollouts in S:
    print('\n\n\n------ num rollouts = '+str(num_rollouts)+' ------')
    if delta==0.01:
        if num_rollouts==8:
            FILE_NAME = 'PerfBoost_10_21_09_58_37'
        elif num_rollouts==16:
            FILE_NAME = 'PerfBoost_10_21_09_57_40'
        elif num_rollouts==32:
            FILE_NAME = 'PerfBoost_10_21_09_54_20'
        elif num_rollouts==64:
            FILE_NAME = 'PerfBoost_10_21_09_59_53'
        elif num_rollouts==128:
            FILE_NAME = 'PerfBoost_10_21_10_00_19'
        elif num_rollouts==256:
            FILE_NAME = 'PerfBoost_10_21_10_09_31'
        elif num_rollouts==512:
            FILE_NAME = 'PerfBoost_10_21_10_27_43'
        else:
            print(num_rollouts)

    # ----- Load -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
    save_folder = os.path.join(save_path, TRAIN_METHOD, cont_type+'_'+now)

    # load training setup
    setup_loaded = torch.load(
        os.path.join(save_path, 'normflow', FILE_NAME, 'setup'),
        map_location=torch.device('cpu')
    )
    assert num_rollouts==setup_loaded['num_rollouts']
    assert delta==setup_loaded['delta']

    # load trained nfm
    nfm_loaded = torch.load(
        os.path.join(save_path, 'normflow', FILE_NAME, 'final_nfm'),
        map_location=torch.device('cpu')
    )
    nfm_keys = nfm_loaded.keys()

    # ------------ 1. Dataset ------------
    dataset = RobotsDataset(
        random_seed=setup_loaded['random_seed'], horizon=setup_loaded['horizon'],
        std_ini=setup_loaded['std_init_plant'], n_agents=2
    )
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=setup_loaded['num_rollouts'], num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)
    # data for plots
    t_ext = setup_loaded['horizon'] * 4
    # plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
    # plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
    # plot_data = plot_data.to(device)
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
            flows += [MyPlanar((num_params,), act=ACT)]     # IMPORTANT: changed the activation function
            for flow_key in ['u', 'w', 'b']:
                nfm_loaded['flows.'+str(flow_num)+'.'+flow_key] = nfm_loaded['flows.'+str(flow_num)+'.'+flow_key].reshape(
                    getattr(flows[flow_num], flow_key).shape
                )
        else:
            raise NotImplementedError

    # init dummy NF model
    nfm = NormalizingFlow(q0=q0, flows=flows, p=target) # NOTE: set back to nf.NormalizingFlow
    # load state dict
    # nfm.load_state_dict(nfm_loaded)
    nfm.to(device)  # Move model on GPU if available
    # for flow in nfm.flows:
    #     print(flow.act)


    # ------ test ------
    num_samples = 1
    sample_from = 'nf_post'
    # sample controllers
    if sample_from=='uniform':
        raise NotImplementedError
        # sampled_inds = random_state.choice(
        #     len(res_dict['posterior']), num_samples, replace=True
        # )
    elif sample_from=='nf_post':
        sampled_controllers, _ = nfm.sample(num_samples)
    elif sample_from=='prior':
        sampled_controllers = gibbs_posteior.prior.sample(torch.Size([num_samples]))
    else:
        raise NotImplementedError
    if len(sampled_controllers.shape)==1:
        sampled_controllers = sampled_controllers.reshape(1, sampled_controllers.shape[0])


    # TEST 1:check new and old implementations of inverse for leaky_relu
    if ACT=='leaky_relu':
        z = sampled_controllers.detach().clone()
        z_old = sampled_controllers.detach().clone()
        for i in range(len(nfm.flows) - 1, -1, -1):
            z_old, log_det_old = nfm.flows[i].old_inverse(z_old)
            z, log_det = nfm.flows[i].inverse(z)
            av_abs_diff = torch.sum(torch.sum(torch.abs(z-z_old)))/z.shape[0]/z.shape[1]
            assert av_abs_diff<=1e-6, 'z mistmatch in flow ' + str(i) + ' with av abs difference' + str(av_abs_diff)
            av_abs_diff = torch.sum(torch.abs(log_det-log_det_old))/log_det.shape[0]
            assert av_abs_diff<=1e-6, 'log_det mistmatch in flow ' + str(i) + ' with av abs difference' + str(av_abs_diff)
        print('Test new vs old inverse was successful.')

    # TEST 2: forward and inverse should cancel each other
    z = sampled_controllers.detach().clone()
    # z_old = sampled_controllers.detach().clone()
    for i in range(len(nfm.flows) - 1, -1, -1):
        print(z.shape)
        z_frw, _ = nfm.flows[i].forward(z)
        print(z_frw.shape)
        z_frw_inv, _ = nfm.flows[i].inverse(z_frw)
        print(z_frw_inv.shape)
        av_abs_diff = torch.sum(torch.sum(torch.abs(z-z_frw_inv)))/z.shape[0]/z.shape[1]
        assert av_abs_diff<=1e-6, 'z != inverse(forward(z)) in flow ' + str(i) + ' with av abs difference' + str(av_abs_diff)
        # for testing the next layer
        z, _ = nfm.flows[i].inverse(z)
    print('Test new vs old inverse was successful.')



