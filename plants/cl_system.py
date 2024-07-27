import torch
from numpy import random

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
