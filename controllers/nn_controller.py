import torch
from collections import OrderedDict

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

        # set number of trainable params
        self.num_params = sum([p.nelement() for p in self.parameters()])

        # convert param to buffer if not empirical
        if not self.train_method=='empirical':
            for i, size in enumerate(layer_sizes):
                layer = getattr(self, 'fc_%i'%(i+1))
                old_weight, old_bias = layer.weight, layer.bias
                del layer.weight
                del layer.bias
                layer.register_buffer('weight', old_weight)
                layer.register_buffer('bias', old_bias)
            # output layer
            layer = getattr(self, 'out')
            old_weight, old_bias = layer.weight, layer.bias
            del layer.weight
            del layer.bias
            layer.register_buffer('weight', old_weight)
            layer.register_buffer('bias', old_bias)

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

