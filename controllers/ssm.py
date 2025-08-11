import math
import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from controllers.non_linearities import MLP, HamiltonianSIE, CouplingLayer


class LRU(nn.Module):
    """
    Implements a Linear Recurrent Unit (LRU) following the parametrization of "Resurrecting Linear Recurrences" paper.
    The LRU is simulated using Parallel Scan (fast!) when scan=True (default), otherwise recursively (slow)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 state_features: int,
                 scan: bool = True,  # This has been removed
                 rmin: float = 0.99,
                 rmax: float = 1.,
                 max_phase: float = 6.283,
                 internal_state_init=None
                 ):
        super().__init__()

        # set dimensions
        self.dim_internal = state_features
        self.dim_in = in_features
        self.scan = scan
        self.dim_out = out_features

        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(lambda_mod) - torch.square(lambda_mod))))
        self.B_real = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))
        self.B_imag = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))
        self.C_real = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))
        self.C_imag = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))
        # self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

        # define trainable params
        self.training_param_names = ['D', 'nu_log', 'theta_log', 'gamma_log', 'B_real', 'B_imag', 'C_real', 'C_imag']

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.complex(torch.zeros(state_features), torch.zeros(state_features))
            # torch.zeros(1, 1, self.dim_internal)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            assert internal_state_init.dim == 1 and internal_state_init.shape[0] == 4
            if internal_state_init.is_complex():
                self.x = self.dim_internal
            else:
                self.x = torch.complex(internal_state_init, torch.zeros(self.dim_internal))
        self.register_buffer('init_x', self.x.detach().clone())

    def forward(self, u_in):
        """
        Forward pass of SSM.
        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        batch_size = u_in.shape[0]

        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.theta_log))
        lambda_c = torch.complex(lambda_re, lambda_im)  # A matrix
        gammas = torch.exp(self.gamma_log)
        print(
            'u_in', u_in.shape,
            'u_in comp', torch.complex(u_in, torch.zeros(1, device=self.B_real.device)).shape,
            'B real', self.B_real.shape,
            'B comp', torch.complex(self.B_real, self.B_imag).shape)
        exit()
        self.x = lambda_c * self.x + gammas * torch.matmul(
            torch.complex(u_in, torch.zeros(1, device=self.B_real.device)), 
            torch.complex(self.B_real, self.B_imag).transpose(-1,-2)
            )
        y_out = 2 * torch.matmul(
            self.x, 
            torch.complex(self.C_real, self.C_imag).transpose(-1, -2)
        ).real + torch.matmul(
            u_in, 
            self.D.transpose(-1, -2)
        )
        return y_out
    
    def get_parameters_as_vector(self):
        """
        Returns the parameters of the LRU as a vector.
        """
        vec = None
        for name in self.training_param_names:
            vec_name = getattr(self, name)
            if name in ['B', 'C']:
                vec_name = torch.cat((vec_name.real.flatten(), vec_name.imag.flatten()), 0)
            if vec is None:
                vec = vec_name.flatten()
            else:
                vec = torch.cat((vec, vec_name.flatten()), 0)
        # TODO: remove sanity checks
        assert vec.requires_grad and not vec.is_leaf, "The vector of parameters must be a leaf tensor with requires_grad=True"
        assert vec.dtype == torch.float32, "The vector of parameters must be of type torch.float32"
        return vec


# Class for implementing LRU + a user-defined scaffolding, this is our SSM block.
class SSM(nn.Module):
    # Scaffolding can be modified. In this case we have LRU, MLP plus linear skip connection.
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_internal: int,
                 dim_scaffolding: int = 30,
                 scan: bool = False,
                 rmin: float = 0.95,
                 rmax: float = 0.99,
                 max_phase: float = 6.283,
                 internal_state_init=None,
                 scaffolding_nonlin: str = "MLP"
                 ):
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_scaffolding = dim_scaffolding

        if scaffolding_nonlin == "MLP":
            self.scaffold = MLP(dim_out, dim_scaffolding, dim_out).to(device)
        elif scaffolding_nonlin == "coupling_layers":
            # Option 2: coupling (or invertible) layers
            self.scaffold = CouplingLayer(dim_out, dim_scaffolding).to(device)
        elif scaffolding_nonlin == "hamiltonian":
            # Option 3: Hamiltonian net
            self.scaffold = HamiltonianSIE(n_layers=4, nf=dim_out, bias=False).to(device)
        elif scaffolding_nonlin == "tanh":
            self.scaffold = torch.tanh.to(device)
        else:
            # End options
            raise NotImplementedError("The scaffolding_nonlin %s is not implemented" % scaffolding_nonlin)
    
        self.lru = LRU(dim_in, dim_out, dim_internal, scan, rmin, rmax, max_phase, internal_state_init).to(device)
        self.lin = nn.Linear(dim_in, dim_out, bias=False).to(device)

        nn.init.zeros_(self.lin.weight.data)

        self.extract_param_names()  # Initialize param_names

    def forward(self, u):
        result = self.scaffold(self.lru(u)) + self.lin(u)
        return result

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, self.state_dict()[name]) for name in self.training_param_names
        )
        return param_dict

    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, self.state_dict()[name].shape) for name in self.training_param_names
        )
        return param_dict
    
    def extract_param_names(self):
        """
        Extracts the names of the parameters in the SSM.
        """
        self.training_param_names = ['lru.' + name for name in self.lru.training_param_names] + \
                                    ['scaffold.' + name for name in self.scaffold.training_param_names] + \
                                    ['lin.weight']
        if self.lin.bias is not None:
            if self.lin.bias.requires_grad:
                self.training_param_names.append('lin.bias')
                
    def get_parameters_as_vector(self):
        vec = torch.cat((self.lru.get_parameters_as_vector(), self.scaffold.get_parameters_as_vector(), self.lin.weight.flatten()), 0)
        if self.lin.bias is not None and self.lin.bias.requires_grad:
            vec = torch.cat((vec, self.lin.bias.flatten()), 0)
        return vec
    
    def _get_tensor_from_name(self, param_name):
        """
        Returns the tensor corresponding to the parameter name.
        """
        if param_name.startswith('lru.'):
            p = getattr(self.lru, param_name[4:])
        elif param_name.startswith('scaffold.'):
            p = self.scaffold._get_tensor_from_name(param_name[9:])
        elif param_name == 'lin.weight':
            p = getattr(self.lin, 'weight')
        elif param_name == 'lin.bias':
            p = getattr(self.lin, 'bias')
        else:
            raise ValueError(f'Unknown parameter name: {param_name}')
        return p

    def set_parameter(self, param_name, value):
        """
        Sets the tensor corresponding to the parameter name.
        """
        if param_name.startswith('lru.'):
            value = torch.nn.Parameter(value)
            setattr(self.lru, param_name[4:], value)
        elif param_name.startswith('scaffold.'):
            value = torch.nn.Parameter(value)
            self.scaffold.set_parameter(param_name[9:], value)
        elif param_name == 'lin.weight':
            value = torch.nn.Parameter(value)
            setattr(self.lin, 'weight', value)
        elif param_name == 'lin.bias':
            value = torch.nn.Parameter(value)
            setattr(self.lin, 'bias', value)
        else:
            raise ValueError(f'Unknown parameter name: {param_name}')


# Class implementing a cascade of N SSMs. Linear pre- and post-processing can be modified
class DeepSSM(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_internal: int,
                 dim_middle: int,
                 dim_scaffolding: int = 30,
                 scan: bool = False,
                 # n_ssm: int,
                 rmin: float = 0.9,
                 rmax: float = 1,
                 max_phase: float = 6.283,
                 internal_state_init=None,
                 scaffolding_nonlin="MLP"
                 ):
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_scaffolding = dim_scaffolding

        self.ssm1 = SSM(
            dim_in=dim_in, dim_out=dim_middle, dim_internal=dim_internal, dim_scaffolding=dim_scaffolding, 
            scan=scan, rmin=rmin, rmax=rmax, max_phase=max_phase, scaffolding_nonlin=scaffolding_nonlin,
            internal_state_init=internal_state_init
        )
        self.ssm2 = SSM(
            dim_in=dim_middle, dim_out=dim_out, dim_internal=dim_internal, dim_scaffolding=dim_scaffolding, 
            scan=scan, rmin=rmin, rmax=rmax, max_phase=max_phase, scaffolding_nonlin=scaffolding_nonlin,
            internal_state_init=internal_state_init
        )

        # count number of parameters
        self.training_param_names = ['ssm1.' + name for name in self.ssm1.training_param_names] + \
                                     ['ssm2.' + name for name in self.ssm2.training_param_names]
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, u_in):
        y_out = self.ssm2(self.ssm1(u_in))
        return y_out

    def reset(self):
        self.ssm1.lru.x = self.ssm1.lru.init_x  # reset the SSM.LRU state to the initial value
        self.ssm2.lru.x = self.ssm2.lru.init_x  # reset the SSM.LRU state to the initial value

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, self.state_dict()[name].shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, self._get_tensor_from_name(name)) for name in self.training_param_names
        )
        return param_dict
    
    def get_parameters_as_vector(self):
        vec = torch.cat((self.ssm1.get_parameters_as_vector(), self.ssm2.get_parameters_as_vector()), 0)
        assert vec.shape[0] == self.num_params
        assert vec.requires_grad and not vec.is_leaf
        return vec
    
    def _get_tensor_from_name(self, param_name):
        if param_name.startswith('ssm1.'):
            p = self.ssm1._get_tensor_from_name(param_name[5:])
        elif param_name.startswith('ssm2.'):
            p = self.ssm2._get_tensor_from_name(param_name[5:])
        else:
            raise ValueError(f'Unknown parameter name: {param_name}')
        return p
    
    def set_parameter(self, param_name, value):
        """
        Sets the tensor corresponding to the parameter name.
        """
        if param_name.startswith('ssm1.'):
            self.ssm1.set_parameter(param_name[5:], value)
        elif param_name.startswith('ssm2.'):
            self.ssm2.set_parameter(param_name[5:], value)
        else:
            raise ValueError(f'Unknown parameter name: {param_name}')

if __name__ == "__main__":
    dim_in = 2
    dim_out = 2
    dim_internal = 4
    dim_scaffolding = 8
    batch_size = 3
    ssm = SSM(dim_in, dim_out, dim_internal, scan=False, dim_scaffolding=dim_scaffolding, scaffolding_nonlin="hamiltonian")
    deep_ssm = DeepSSM(dim_in, dim_out, dim_internal, dim_middle=6, dim_scaffolding=dim_scaffolding, scaffolding_nonlin="hamiltonian")
    
    # Print dimensions:
    print("B has dimensions: ", ssm.lru.B.shape)
    print("C has dimensions: ", ssm.lru.C.shape)
    print("D has dimensions: ", ssm.lru.D.shape)
    print("nu_log has dimensions: ", ssm.lru.nu_log.shape)
    print("theta_log has dimensions: ", ssm.lru.theta_log.shape)
    print("gamma_log has dimensions: ", ssm.lru.gamma_log.shape)
    print("the state has dimensions: ", ssm.lru.x.shape)

    # # Test methods
    # param_dict = ssm.get_named_parameters()
    # print(param_dict)

    t = torch.linspace(0, 1, 100)
    u = torch.zeros(batch_size, t.shape[0], dim_in)
    for i in range(batch_size):
        u[i, 0, :] = torch.randn(dim_in)
    y_ssm = torch.zeros(batch_size, t.shape[0], dim_out)
    y_deep_ssm = torch.zeros(batch_size, t.shape[0], dim_out)
    x_ssm = torch.complex(torch.zeros(batch_size, t.shape[0], dim_internal), torch.zeros(1, t.shape[0], dim_internal))
    x_deep_ssm = torch.complex(torch.zeros(batch_size, t.shape[0], dim_internal),
                               torch.zeros(1, t.shape[0], dim_internal))
    for i in range(t.shape[0]):
        y_ssm[:, i:i + 1, :] = ssm(u[:, i:i + 1, :])
        x_ssm[:, i:i + 1, :] = ssm.lru.x
        y_deep_ssm[:, i:i + 1, :] = deep_ssm(u[:, i:i + 1, :])
        x_deep_ssm[:, i:i + 1, :] = deep_ssm.ssm1.lru.x

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t, y_ssm[0, :, :].detach())
    plt.title("Output SSM")
    plt.figure()
    plt.plot(t, u[0, :, :].detach())
    plt.title("Input")
    plt.figure()
    plt.plot(t, x_ssm[0, :, :].real.detach())
    plt.title("State SSM")
    plt.figure()
    plt.plot(t, y_deep_ssm[0, :, :].detach())
    plt.title("Output Deep SSM")
    plt.figure()
    plt.plot(t, x_deep_ssm[0, :, :].real.detach())
    plt.title("State 1st SSM of Deep SSM")
    plt.show()
