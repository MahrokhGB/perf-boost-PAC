import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger

TRAIN_METHOD = 'empirical'

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'empirical')
save_folder = os.path.join(save_path, 'perf_boost_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)
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
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

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
    ).to(device)
elif args.cont_type=='NN':
    ctl_generic = NNController(
        in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=args.layer_sizes,
        train_method=TRAIN_METHOD
    ).to(device)
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

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl_generic.parameters(), lr=args.lr)

# ------------ 6. Training ------------
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=dataset.xbar, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_init.pdf',
    text="CL - before training", T=t_ext,
    obstacle_centers=original_loss_fn.obstacle_centers,
    obstacle_covs=original_loss_fn.obstacle_covs
)
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl_generic, data=train_data_batch
        )
        # loss of this rollout
        loss = original_loss_fn.forward(x_log, u_log)
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f'% (epoch, loss)
        print(ctl_generic.get_parameters_as_vector()[0:10])

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl_generic, data=valid_data
                )
                # loss of the valid data
                original_loss_valid = original_loss_fn.forward(x_log_valid, u_log_valid)
                bounded_loss_valid = bounded_loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.2f' % (original_loss_valid.item())
            msg += ' ---||--- bounded validation loss: %.2f' % (bounded_loss_valid.item())
            # compare with the best valid loss
            if original_loss_valid.item()<best_valid_loss:
                best_valid_loss = original_loss_valid.item()
                best_params = ctl_generic.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl_generic.set_parameters_as_vector(best_params)

# ------ 7. Save and evaluate the trained model ------
# save
res_dict = ctl_generic.c_ren.state_dict()
res_dict['Q'] = Q
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl_generic, data=train_data
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = original_loss_fn.forward(x_log, u_log)
    msg = 'Loss: %.4f' % (loss)
# count collisions
if args.col_av:
    num_col = original_loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl_generic, data=test_data
    )
    # loss
    test_loss = original_loss_fn.forward(x_log, u_log).item()
    msg = "Loss: %.4f" % (test_loss)
# count collisions
if args.col_av:
    num_col = original_loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, _, u_log = sys.rollout(ctl_generic, plot_data)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=dataset.xbar, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_trained.pdf',
    text="CL - trained controller", T=t_ext,
    obstacle_centers=original_loss_fn.obstacle_centers,
    obstacle_covs=original_loss_fn.obstacle_covs
)