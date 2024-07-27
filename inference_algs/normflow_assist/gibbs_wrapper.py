import torch
from normflows.distributions import Target

class GibbsWrapperNF(Target):
    """
    Wrap given Gibbs distribution to be used in normflows
    """

    def __init__(
        self, target_dist, train_dataloader,
        prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
    ):
        """Constructor

        Args:
          target_dist: Distribution to be approximated
          train_dataloader: Data loader used to compute the Gibbs dist for training
          prop_scale: Scale for the uniform proposal
          prop_shift: Shift for the uniform proposal
        """
        super().__init__(prop_scale=prop_scale, prop_shift=prop_shift)
        self.target_dist = target_dist
        self.train_data_iterator = list(train_dataloader)
        self.max_log_prob = 0.0
        self.data_batch_ind = 0
        # TODO: random seed must be fixed across REN controller, ...

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        train_data = self.train_data_iterator[self.data_batch_ind]
        self.data_batch_ind = (self.data_batch_ind+1) % len(self.train_data_iterator)
        # t = time.time()
        lp = self.target_dist.log_prob(params=z, train_data=train_data)
        # print('log prob time ', time.time()-t)
        self.train_data = None  # next call requires setting a new training data
        return lp
