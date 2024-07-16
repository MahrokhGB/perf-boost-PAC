import torch, sys
from numpy import random
from config import BASE_DIR
sys.path.append(BASE_DIR)
from config import device


# The `CLSystem` class is a neural network module that performs multi-rollout simulations using a
# given system and controller.
class CLSystem(torch.nn.Module):
    def __init__(self, sys, controller, random_seed):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            self.random_state = random.RandomState(random_seed)
        else:
            self.random_state = random.RandomState(0)
        self.sys=sys
        self.controller=controller

    def rollout(self, data):
        assert len(data.shape)==3
        (S, T, state_dim) = data.shape
        assert state_dim==self.sys.state_dim

        xs, ys, us= self.sys.rollout(
            controller=self.controller,
            data=data, train=True
        )

        assert xs.shape==(S, T, state_dim), xs.shape
        return xs, ys, us

    def forward(self, data):
        xs, ys, us = self.rollout(data)
        return (xs, us)


# from controllers.REN_controller import RENController
# def get_controller(
#     controller_type, sys,
#     # REN controller
#     n_xi=None, l=None, initialization_std=None,
#     train_method='SVGD', output_amplification=20
# ):
#     if controller_type == 'REN':
#         assert not (n_xi is None or l is None)
#         generic_controller = RENController(
#             noiseless_forward=sys.noiseless_forward,
#             output_amplification=output_amplification,
#             state_dim=sys.state_dim, in_dim=sys.in_dim,
#             n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init,
#             train_method=train_method, initialization_std=initialization_std
#         )
#     elif controller_type=='Affine':
#         generic_controller = AffineController(
#             weight=torch.zeros(sys.in_dim, sys.state_dim, device=device, dtype=torch.float32),
#             bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32)
#         )
#     else:
#         raise NotImplementedError

#     return generic_controller

# ---------- CONTROLLER ----------
from collections import OrderedDict
from assistive_functions import to_tensor
class AffineController(torch.nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        # weight is a tensor of shape = (in_dim, state_dim)
        weight = to_tensor(weight)
        if len(weight.shape)==1:
            weight = weight.reshape(1, -1)
        # bias is a tensor of shape=(in_dim, 1)
        if bias is not None:
            bias = to_tensor(bias)
            if len(bias.shape)==1:
                bias = bias.reshape(-1, 1)
        # define params
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_buffer('bias', torch.zeros((weight.shape[0], 1)))

        # check dimensions
        self.in_dim, self.state_dim = self.weight.shape
        assert self.bias.shape==(self.in_dim, 1)


    def forward(self, what):
        # what must be of shape (batch_size, state_dim, self.in_dim)
        # assert what.shape[1:]==torch.Size([self.state_dim, self.in_dim]), what.shape
        if what.shape[-1]==self.state_dim:
            what = what.transpose(-1,-2)
        return torch.matmul(self.weight, what)+self.bias

    def set_parameters_as_vector(self, vec):
        # last element is bias, the rest is weight
        vec = vec.flatten()
        assert len(vec) == sum([p.nelement() for p in self.parameters()])
        self.set_parameter('weight', vec[:self.weight.nelement()])
        self.set_parameter('bias', vec[self.weight.nelement():])

    def set_parameter(self, name, value):
        current_val = getattr(self, name)
        value = torch.nn.Parameter(value.reshape(current_val.shape))
        setattr(self, name, value)

    def reset(self):
        return

    def parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in ['weight', 'bias']
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in ['weight', 'bias']
        )
        return param_dict
