import torch
import torch.nn as nn
import numpy as np

from config import device
from .contractive_ren import ContractiveREN
from controllers.ssm import DeepSSM


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state
        of the plant and the controller output is the input to the plant.
        This controller has a memory for the last input ("self.last_input") and
        the last output ("self.last_output").
    """
    def __init__(self,
                 noiseless_forward,
                 input_init: torch.Tensor,
                 output_init: torch.Tensor,
                 nn_type: str = "REN",
                 dim_internal: int = 8,
                 output_amplification: float = 20,
                 train_method: str = 'empirical',
                 # SSM properties
                 scaffolding_nonlin: str = None,
                 dim_middle: int = 6,
                 dim_scaffolding: int = 30,
                 rmin: float = 0.9,
                 rmax: float = 1.0,
                 max_phase: float = 6.283,
                 # acyclic REN properties
                 dim_nl: int = 8,
                 initialization_std: float = 0.5,
                 pos_def_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0,
                 ren_internal_state_init=None,
                 ):
        """
         Args:
            noiseless_forward:            System dynamics without process noise. It can be TV.
            input_init (torch.Tensor):    Initial input to the controller.
            output_init (torch.Tensor):   Initial output from the controller before anything is calculated.
            nn_type (str):                Which NN model to use for the Emme operator (Options: 'REN' or 'SSM')
            dim_internal (int):           Internal state (x) dimension.
            output_amplification (float): Scaling factor applied to the controller output. Default is 20.
            train_method (str): Training method. Defaults to empirical
            ##### SSM-specific args:
            scaffolding_nonlin (str):     Non-linearity used in SSMs for scaffolding.
            dim_middle (int):             [Optional] Middle dimension for SSM deep architecture. Default is 6.
            dim_scaffolding (int):        [Optional] Dimension of the hidden layers of scaffolding for SSM architecture. Only used for MLP and coupling_layers scaffolding. Default is 30.
            rmin (float):                 [Optional] Minimum radius for SSM LRU initialization. Default is 0.9.
            rmax (float):                 [Optional] Maximum radius for SSM LRU initialization. Default is 1.0.
            max_phase (float):            [Optional] Maximum phase for SSM LRU initialization. Default is 6.283.
            ##### REN-specific args:
            dim_nl (int):                 Dimension of the input ("v") and output ("w") of the NL static block of REN.
            initialization_std (float):   [Optional] Weight initialization. Set to 0.1 by default.
            pos_def_tol (float):          [Optional] Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float):  [Optional] Lower bound on the contraction rate. Default to 1.
            ren_internal_state_init (torch.Tensor): [Optional] Initial state of the REN. Default to 0 when None.
        """
        super().__init__()

        self.output_amplification = output_amplification
        self.train_method = train_method

        # set initial conditions
        self.input_init = input_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        # set dimensions
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]

        # set type of nn for emme
        self.nn_type = nn_type
        # define Emme as REN or SSM
        if nn_type == "REN":
            self.emme = ContractiveREN(
                dim_in=self.dim_in, dim_out=self.dim_out, dim_internal=dim_internal,
                dim_nl=dim_nl, initialization_std=initialization_std,
                internal_state_init=ren_internal_state_init,
                pos_def_tol=pos_def_tol, contraction_rate_lb=contraction_rate_lb
            ).to(device)
        elif nn_type == "SSM":
            # define the SSM
            self.emme = DeepSSM(
                dim_in=self.dim_in,
                dim_out=self.dim_out,
                dim_internal=dim_internal,
                dim_middle=dim_middle,
                dim_scaffolding=dim_scaffolding,
                scan=False,
                rmin=rmin,
                rmax=rmax,
                max_phase=max_phase,
                internal_state_init=None,
                scaffolding_nonlin=scaffolding_nonlin
            ).to(device)
        else:
            raise ValueError("Model for emme not implemented")

        # set number of trainable params
        self.num_params = self.emme.num_params

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward

        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.emme.reset()    # reset the REN state to the initial value

    def forward(self, input_t: torch.Tensor):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # assert self.emme.X.requires_grad
        # apply noiseless forward to get noise less input (noise less state of the plant)
        u_noiseless = self.noiseless_forward(
            t=self.t,
            x=self.last_input,  # last input to the controller is the last state of the plant
            u=self.last_output  # last output of the controller is the last input to the plant
        )  # shape = (self.batch_size, 1, self.dim_in)

        # reconstruct the noise
        w_ = input_t - u_noiseless # shape = (self.batch_size, 1, self.dim_in)

        # apply REN
        output = self.emme.forward(w_)
        output = output*self.output_amplification   # shape = (self.batch_size, 1, self.dim_out)

        # update internal states
        self.last_input, self.last_output = input_t, output
        self.t += 1
        return output

    # setters and getters
    def get_parameter_shapes(self):
        return self.emme.get_parameter_shapes()

    def get_named_parameters(self):
        return self.emme.get_named_parameters()

    # # def get_parameters_as_vector(self):
    # #     # TODO: implement without numpy
    # #     return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.emme.parameters()])

    def set_parameter(self, name, value):
        # param_shape = getattr(self.emme, name+'_shape')
        param_shape = self.emme.get_parameter_shapes()[name]
        if torch.empty(param_shape).nelement()==value.nelement():
            value = value.reshape(param_shape)
        else:
            value = value.reshape(value.shape[0], *param_shape)
        if self.train_method=='empirical':
            value = torch.nn.Parameter(value)
        if isinstance(self.emme, DeepSSM):
            # set the parameter in the SSM
            self.emme.set_parameter(name, value)
        else:
            setattr(self.emme, name, value)
            self.emme._update_model_param()    # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        # flatten vec if not batched
        if value.nelement()==self.num_params:
            value = value.flatten()
            # batched = False
        # else:
        #     batched = True
        # value is reshaped to the parameter shape
        idx = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape[0]
            else:
                dim = shape[-1]*shape[-2]
            # elif len(shape) == 2:
            #     dim = shape[0]*shape[1]
            # else:
            #     raise NotImplementedError
            idx_next = idx + dim
            # select indx
            if value.ndim == 1:
                value_tmp = value[idx:idx_next]
            elif value.ndim == 2:
                value_tmp = value[:, idx:idx_next]
            elif value.ndim == 3:
                value_tmp = value[:, :, idx:idx_next]
            else:
                raise AssertionError
            # set
            if self.train_method in ['SVGD', 'normflow']:
                self.set_parameter(name, value_tmp)
            elif self.train_method=='empirical':
                with torch.no_grad():
                    self.set_parameter(name, value_tmp)
            else:
                raise NotImplementedError
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return list(self.get_named_parameters().values())

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def get_parameters_as_vector(self):
        return self.emme.get_parameters_as_vector()