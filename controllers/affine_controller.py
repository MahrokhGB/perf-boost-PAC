import torch
from collections import OrderedDict
from utils.assistive_functions import to_tensor

class AffineController(torch.nn.Module):
    def __init__(self, weight, train_method, bias=None):
        super().__init__()

        assert train_method in ['empirical', 'normflow', 'SVGD']
        self.train_method = train_method

        # set number of trainable params
        self.weight_nelement = weight.nelement()
        self.bias_nelement = 0 if bias is None else bias.nelement()
        self.num_params = self.weight_nelement + self.bias_nelement

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
        return torch.matmul(self.weight, what.transpose(-1,-2))+self.bias

    def set_parameters_as_vector(self, vec):
        # last element is bias, the rest is weight
        # check if is batched
        if vec.ndim==self.num_params:
            vec = vec.flatten()
            new_weight = vec[:self.weight_nelement]
            new_bias = vec[self.weight_nelement:]
        else:
            new_weight = vec[:, :self.weight_nelement]
            new_bias = vec[:, self.weight_nelement:]

        self.set_parameter('weight', new_weight)
        self.set_parameter('bias', new_bias)

    def set_parameter(self, name, value):
        current_val = getattr(self, name)
        if value.nelement() == current_val.nelement():
            value = value.reshape(current_val.shape)
        else:
            value = value.reshape(value.shape[0], *current_val.shape)
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

    def get_parameters_as_vector(self):
        return [self.weight.flatten(), self.bias.flatten()]
