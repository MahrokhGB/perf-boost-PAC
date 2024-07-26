import torch
from utils.assistive_functions import to_tensor

# ---------- SYSTEM ----------
class LTISystem(torch.nn.Module):
    def __init__(self, A, B, C, x_init, u_init=None):
        """

        Args:
            x_init: initial state of the plant.
            u_init: initial input to the plant. Defaults to zero when None.
        """

        super().__init__()

        # set matrices
        self.register_buffer('A', to_tensor(A))
        self.register_buffer('B', to_tensor(B))
        self.register_buffer('C', to_tensor(C))

        # Dimensions
        self.state_dim = self.A.shape[0]
        self.in_dim = self.B.shape[1]
        self.out_dim = self.C.shape[0]
        # Check matrices
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, self.in_dim)
        assert self.C.shape == (self.out_dim, self.state_dim)

        # initial state
        self.register_buffer('x_init', to_tensor(x_init).reshape(1, self.state_dim)) # shape = (1, state_dim)
        u_init = torch.zeros(1, self.in_dim) if u_init is None else to_tensor(u_init).reshape(1, self.in_dim)   # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)

    # # simulation
    # def rollout(self, controller, data: torch.Tensor):
    #     """
    #     rollout with state-feedback controller

    #     Args:
    #         - controller: state-feedback controller
    #         - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
    #     """
    #     xs = (data[:, 0:1, :] + self.x_init)
    #     us = controller.forward(xs[:, 0:1, :])
    #     ys = torch.matmul(self.C, xs[:, 0:1, :])
    #     for t in range(1, data.shape[1]):
    #         xs = torch.cat(
    #             (
    #                 xs,
    #                 torch.matmul(self.A, xs[:, t-1:t, :]) + torch.matmul(self.B, us[:, t-1:t, :]) + data[:, t:t+1, :]),
    #             1
    #         )
    #         ys = torch.cat(
    #             (ys, torch.matmul(self.C, xs[:, t:t+1, :])),
    #             1
    #         )
    #         us = torch.cat(
    #             (us, controller.forward(xs[:, t:t+1, :])),
    #             1
    #         )
    #     if not train:
    #         xs, ys, us = xs.detach(), ys.detach(),us.detach()

    #     return xs, ys, us

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)
        f = torch.matmul(self.A, x) + torch.matmul(self.B, u)
        return f    # shape = (batch_size, 1, state_dim)

    def rollout(self, controller, data):
        """
        rollout REN for rollouts of the process noise

        Args:
            - data: sequence of disturbance samples of shape
                (batch_size, T, state_dim).

        Rerurn:
            - x_log of shape = (batch_size, T, state_dim)
            - u_log of shape = (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)

        # Simulate
        for t in range(data.shape[1]):
            x = self.forward(t=t, x=x, u=u, w=data[:, t:t+1, :])    # shape = (batch_size, 1, state_dim)
            u = controller(x)                                       # shape = (batch_size, 1, in_dim)

            if t == 0:
                x_log, u_log = x, u
            else:
                x_log = torch.cat((x_log, x), 1)
                u_log = torch.cat((u_log, u), 1)

        controller.reset()

        return x_log, None, u_log

    def forward(self, t, x, u, w):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """
        return self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)