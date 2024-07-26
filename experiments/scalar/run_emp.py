import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from nf_assistive_functions import eval_norm_flow
from arg_parser import argument_parser, print_args
from plants import LTISystem, LTIDataset
# from plot_functions import *
from controllers import AffineController, NNController, PerfBoostController
from loss_functions import LQLossFH
from utils.assistive_functions import WrapLogger, sample_2d_dist

import numpy as np #TODO: remove
from control import dlqr
import matplotlib.pyplot as plt
import pickle

PLOT_DIST = True
BASE_IS_PRIOR = False

# ----- parse and set experiment arguments ----- TODO
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
logger.info('------------ EMPIRICAL ------------')

logger.info(msg)
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
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

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
        bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32)
    )
elif args.cont_type=='NN':
    ctl_generic = NNController(
        in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=[32,32]
    )
else:
    raise KeyError('[Err] args.cont_type must be PerfBoost, Affine, or NN.')
num_params = sum([p.nelement() for p in ctl_generic.parameters()])
logger.info('[INFO] Controller is of type ' + args.cont_type + ' and has %i parameters.' % num_params)
logger.info(ctl_generic.get_parameters_as_vector())

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

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl_generic.parameters(), lr=args.lr)

# ------------ 6. Training ------------
with torch.no_grad():
    x_log_valid, _, u_log_valid = sys.rollout(
        controller=ctl_generic, data=valid_data
    )
    # loss of the valid data
    loss_valid = bounded_loss_fn.forward(x_log_valid, u_log_valid)
logger.info('initial validation loss: %.2f' % (loss_valid.item()))

logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
param_history = [ctl_generic.get_parameters_as_vector()]
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl_generic, data=train_data_batch
        )
        # loss of this rollout
        loss = bounded_loss_fn.forward(x_log, u_log)
        # take a step
        loss.backward()
        optimizer.step()
    param_history.append(ctl_generic.get_parameters_as_vector())
    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f'% (epoch, loss)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl_generic, data=valid_data
                )
                # loss of the valid data
                loss_valid = bounded_loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = ctl_generic.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl_generic.set_parameters_as_vector(torch.Tensor(best_params))

# ------ 7. Save and evaluate the trained model ------
# save
res_dict = ctl_generic.c_ren.state_dict()
# TODO: append args
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
    loss = bounded_loss_fn.forward(x_log, u_log)
    msg = 'Loss: %.4f' % (loss)
logger.info(msg)

logger.info(ctl_generic.get_parameters_as_vector())

param_history = np.array(param_history)
fig, ax = plt.subplots(1,1,figsize=(10,10))
for ind in range(num_params):
    params = param_history[:, ind]
    ax.plot(range(len(params)), params, label='param %i'%ind)
plt.legend()
plt.savefig(os.path.join(save_folder, 'paramhistory.pdf'))
plt.show()