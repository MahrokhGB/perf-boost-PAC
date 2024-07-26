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
            data=data
        )

        # assert xs.shape==(S, T, state_dim), xs.shape
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
from utils.assistive_functions import to_tensor
class AffineController(torch.nn.Module):
    def __init__(self, weight, train_method, bias=None):
        super().__init__()

        assert train_method in ['empirical', 'normflow', 'SVGD']
        self.train_method = train_method

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
        if train_method=='empirical':
            self.weight = torch.nn.Parameter(weight)
        else:
            self.register_buffer('weight', weight)
        if bias is not None:
            if train_method=='empirical':
                self.bias = torch.nn.Parameter(bias)
            else:
                self.register_buffer('bias', bias)
        else:
            self.register_buffer('bias', torch.zeros((weight.shape[0], 1)))

        # check dimensions
        self.out_dim, self.in_dim = self.weight.shape
        assert self.bias.shape==(self.out_dim, 1)


    def forward(self, what):
        # what must be of shape (batch_size, state_dim, self.in_dim)
        # assert what.shape[1:]==torch.Size([self.state_dim, self.in_dim]), what.shape
        if what.shape[-1]==self.in_dim:
            what = what.transpose(-1,-2)
        return torch.matmul(self.weight, what)+self.bias

    def set_parameters_as_vector(self, vec):
        # last element is bias, the rest is weight
        vec = vec.flatten()
        # assert len(vec) == sum([p.nelement() for p in self.parameters()]) # TODO
        self.set_parameter('weight', vec[:self.weight.nelement()])
        self.set_parameter('bias', vec[self.weight.nelement():])

    def set_parameter(self, name, value):
        current_val = getattr(self, name)
        value = value.reshape(current_val.shape)
        if self.train_method=='empirical':
            value = torch.nn.Parameter(value)
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



class NNController(torch.nn.Module):
    def __init__(self, train_method, in_dim, out_dim, layer_sizes, nonlinearity_hidden=torch.tanh, nonlinearity_output=None):
        super(NNController, self).__init__()

        assert train_method in ['empirical', 'normflow', 'SVGD']
        self.train_method = train_method
        self.in_dim, self.out_dim = in_dim, out_dim
        self.nonlinearity_hidden, self.nonlinearity_output = nonlinearity_hidden, nonlinearity_output

        self.n_layers = len(layer_sizes)
        self.layers = []
        prev_size = in_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, 'fc_%i'%(i+1), torch.nn.Linear(prev_size, size))
            prev_size = size
        setattr(self, 'out', torch.nn.Linear(prev_size, self.out_dim))
        # TODO: if not empirical, del params

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers+1):
            output = getattr(self, 'fc_%i'%i)(output)
            if self.nonlinearity_hidden is not None:
                output = self.nonlinearity_hidden(output)
        output = getattr(self, 'out')(output)
        if self.nonlinearity_output is not None:
            output = self.nonlinearity_output(output)
        return output

    def forward_parametrized(self, x, params):
        output = x
        param_idx = 0
        for i in range(1, self.n_layers + 1):
            output = torch.nn.functional.linear(output, params[param_idx], params[param_idx+1])
            output = self.nonlinearlity(output)
            param_idx += 2
        output = torch.nn.functional.linear(output, params[param_idx], params[param_idx+1])
        return output

    def set_parameters_as_vector(self, vec):
        vec = vec.flatten()
        ind=0

        # --- set params of the hidden layers ---
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'fc_%i'%(i))
            # get old weight and bias
            old_weight = getattr(layer, 'weight')
            old_bias = getattr(layer,'bias')
            # get new weight and bias
            new_weight = vec[ind:ind+old_weight.nelement()].reshape(old_weight.shape)
            ind = ind+old_weight.nelement()
            new_bias = vec[ind:ind+old_bias.nelement()].reshape(old_bias.shape)
            ind = ind+old_bias.nelement()
            # convert to param if needed
            if self.train_method == 'empirical':
                new_weight = torch.nn.Parameter(new_weight)
                new_bias = torch.nn.Parameter(new_bias)
            # set new weight and bias
            setattr(layer, 'weight', new_weight)
            setattr(layer, 'bias', new_bias)

        # --- set params of the output layer ---
        layer = getattr(self, 'out')
        # get old weight and bias
        old_weight = getattr(layer, 'weight')
        old_bias = getattr(layer, 'bias')
        # get new weight and bias
        new_weight = vec[ind:ind+old_weight.nelement()].reshape(old_weight.shape)
        ind = ind+old_weight.nelement()
        new_bias = vec[ind:ind+old_bias.nelement()].reshape(old_bias.shape)
        ind = ind+old_bias.nelement()
        # convert to param if needed
        if self.train_method == 'empirical':
            new_weight = torch.nn.Parameter(new_weight)
            new_bias = torch.nn.Parameter(new_bias)
        # set new weight and bias
        setattr(layer, 'weight', new_weight)
        setattr(layer, 'bias', new_bias)

        # --- check all params are used ---
        assert ind == len(vec)

    def get_parameters_as_vector(self):
        vec = torch.Tensor([])
        # get params of the hidden layers
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'fc_%i'%(i))
            weight = getattr(layer, 'weight')
            vec = torch.cat((vec, weight.flatten()) , 0)
            bias = getattr(layer, 'bias')
            vec = torch.cat((vec, bias.flatten()) , 0)
        # get params of the output layer
        layer = getattr(self, 'out')
        weight = getattr(layer, 'weight')
        vec = torch.cat((vec, weight.flatten()) , 0)
        bias = getattr(layer, 'bias')
        vec = torch.cat((vec, bias.flatten()) , 0)

        return vec

    def set_parameter(self, name, value):
        current_val = getattr(self, name)
        value = value.reshape(current_val.shape)
        if self.train_method == 'empirical':
            value = torch.nn.Parameter(value)
        setattr(self, name, value)

    def reset(self):
        return

    # def parameter_shapes(self):
    #     param_dict = OrderedDict(
    #         (name, getattr(self, name).shape) for name in ['weight', 'bias']
    #     )
    #     return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict()
        for i in range(1, self.n_layers + 1):
            for name in ['weight', 'bias']:
                param = getattr(
                    getattr(self, 'fc_%i'%(i)),
                    name
                )
                param_dict['fc_%i'%(i)+'.'+name] = param
        return param_dict

