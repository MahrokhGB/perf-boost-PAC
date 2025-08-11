import torch
from collections import OrderedDict

class batched_linear_layer(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        assert not input.isinf().any()
        if not self.bias is None:
            return torch.matmul(input, self.weight.transpose(-1,-2)) + self.bias.unsqueeze(dim=-2)
        else:
            return torch.matmul(input, self.weight.transpose(-1,-2))


class NNController(torch.nn.Module):
    def __init__(self, train_method, in_dim, out_dim, layer_sizes, nonlinearity_hidden=torch.tanh, nonlinearity_output=None):
        super(NNController, self).__init__()

        assert train_method in ['empirical', 'normflow', 'SVGD']
        self.train_method = train_method
        self.in_dim, self.out_dim = in_dim, out_dim
        self.nonlinearity_hidden, self.nonlinearity_output = nonlinearity_hidden, nonlinearity_output

        self.n_layers = len(layer_sizes)
        self.layers = []
        self.weight_shape = [None]*(self.n_layers+1)
        self.bias_shape = [None]*(self.n_layers+1)
        prev_size = in_dim
        # set up hidden layers
        for i, size in enumerate(layer_sizes):
            layer = batched_linear_layer(prev_size, size)
            setattr(self, 'fc_%i'%(i+1), layer)
            prev_size = size
            self.weight_shape[i] = layer.weight.shape
            self.bias_shape[i] = layer.bias.shape
        # set up the output layer
        layer = batched_linear_layer(prev_size, self.out_dim)
        setattr(self, 'out', layer)
        self.weight_shape[self.n_layers] = layer.weight.shape
        self.bias_shape[self.n_layers] = layer.bias.shape

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
        for i in range(self.n_layers):
            output = getattr(self, 'fc_%i'%(i+1))(output)
            if self.nonlinearity_hidden is not None:
                output = self.nonlinearity_hidden(output)
        output = getattr(self, 'out')(output)
        if self.nonlinearity_output is not None:
            output = self.nonlinearity_output(output)
        return output

    def set_parameters_as_vector(self, vec):
        # flatten vec if not batched
        if vec.nelement()==self.num_params:
            vec = vec.flatten()
            batched = False
        else:
            batched = True

        # --- set params of the layers ---
        ind=0
        for i in range(self.n_layers + 1):
            layer = getattr(self, 'out') if i==self.n_layers else getattr(self, 'fc_%i'%(i+1))
            # get new weight and bias
            if not batched:
                ind_nxt = ind+torch.empty(self.weight_shape[i]).nelement()
                new_weight = vec[ind:ind_nxt].reshape(self.weight_shape[i])
                ind = ind_nxt
                ind_nxt = ind+torch.empty(self.bias_shape[i]).nelement()
                new_bias = vec[ind:ind_nxt].reshape(self.bias_shape[i])
                ind = ind_nxt
            else:
                ind_nxt = ind+torch.empty(self.weight_shape[i]).nelement()
                new_weight = vec[:, ind:ind_nxt].reshape(vec.shape[0], *self.weight_shape[i])
                ind = ind_nxt
                ind_nxt = ind+torch.empty(self.bias_shape[i]).nelement()
                new_bias = vec[:, ind:ind_nxt].reshape(vec.shape[0], *self.bias_shape[i])
                ind = ind_nxt

            # convert to param if needed
            if self.train_method == 'empirical':
                new_weight = torch.nn.Parameter(new_weight)
                new_bias = torch.nn.Parameter(new_bias)
            # set new weight and bias
            setattr(layer, 'weight', new_weight)
            setattr(layer, 'bias', new_bias)

        # --- check all params are used ---
        if not batched:
            assert ind == len(vec)
        else:
            assert ind == vec.shape[-1]

    def get_parameters_as_vector(self):
        # get params of the hidden layers
        vec = None
        for i in range(1, self.n_layers + 1):
            layer = getattr(self, 'fc_%i'%(i))
            weight = getattr(layer, 'weight')
            if i==1:
                vec = weight.flatten()
            else:
                vec = torch.cat((vec, weight.flatten()) , 0)
            bias = getattr(layer, 'bias')
            vec = torch.cat((vec, bias.flatten()) , 0)
        # get params of the output layer
        layer = getattr(self, 'out')
        weight = getattr(layer, 'weight')
        if vec is None: # no hidden layers
            vec = weight.flatten()
        else:
            vec = torch.cat((vec, weight.flatten()), 0)
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

