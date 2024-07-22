import torch
import numpy as np
from config import device


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x


class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = (logger is not None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


def sample_2d_dist(dist, num_samples):
    assert isinstance(dist, np.ndarray)
    assert len(dist.shape)==2

    # Create a flat copy of the distribution
    flat = dist.flatten()

    # Sample indices from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=num_samples)

    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, dist.shape)
    if num_samples>1:
        adjusted_index = np.array(list(zip(*adjusted_index)))

    return adjusted_index