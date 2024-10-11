import torch
from . import LQLossFHMultiBatch
from config import device


class RobotsLossMultiBatch(LQLossFHMultiBatch):
    def __init__(
        self, xbar, Q, alpha_u=1,
        alpha_col=None, alpha_obst=None,
        loss_bound=None, sat_bound=None,
        n_agents=2, min_dist=0.5,
        obstacle_centers=None, obstacle_covs=None
    ):
        super().__init__(Q=Q, R=alpha_u, loss_bound=loss_bound, sat_bound=sat_bound, xbar=xbar)
        self.n_agents = n_agents
        self.alpha_col, self.alpha_obst, self.min_dist = alpha_col, alpha_obst, min_dist
        assert (self.alpha_col is None and self.min_dist is None) or not (self.alpha_col is None or self.min_dist is None)
        if self.alpha_col is not None:
            assert self.n_agents is not None
        # define obstacles
        if obstacle_centers is None:
            self.obstacle_centers = [
                torch.tensor([[-2.5, 0]], device=device),
                torch.tensor([[2.5, 0.0]], device=device),
                torch.tensor([[-1.5, 0.0]], device=device),
                torch.tensor([[1.5, 0.0]], device=device),
            ]
        else:
            self.obstacle_centers = obstacle_centers
        if obstacle_covs is None:
            self.obstacle_covs = [
                torch.tensor([[0.2, 0.2]], device=device)
            ] * len(self.obstacle_centers)
        else:
            self.obstacle_covs = obstacle_covs

        # mask
        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))   # shape = (n_agents, n_agents)

    def forward(self, xs, us):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (*batch_shape, T, state_dim)
            - us: tensor of shape (*batch_shape, T, in_dim)

        Return:
            - loss of shape (*batch_shape[0:-1], 1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        # loss states = 1/T sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        if self.xbar is not None:
            x_batch_centered = x_batch - self.xbar
        else:
            x_batch_centered = x_batch
        xTQx = torch.matmul(
            torch.matmul(x_batch_centered.transpose(-1, -2), self.Q),
            x_batch_centered
        )         # shape = (*batch_dim, T, 1, 1)
        loss_x = torch.sum(xTQx, -3) / x_batch.shape[-3]    # average over the time horizon. shape = (*batch_dim, 1, 1)
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )   # shape = (*batch_dim, T, 1, 1)
        loss_u = torch.sum(uTRu, -3) / x_batch.shape[-3]    # average over the time horizon. shape = (*batch_dim, 1, 1)]
        # collision avoidance loss
        if self.alpha_col is None:
            loss_ca = 0
        else:
            loss_ca = self.alpha_col * self.f_loss_ca(x_batch)       # shape = (*batch_dim, 1, 1)
        # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch) # shape = (*batch_dim, 1, 1)
        # sum up all losses
        loss_val = loss_x + loss_u + loss_ca + loss_obst       # shape = (*batch_dim, 1, 1)
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (*batch_dim, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = ((*batch_dim, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, -3)/loss_val.shape[-3]       # shape = (*batch_dim[0:-2], 1, 1)
        return loss_val

    def f_loss_obst(self, x_batch):
        """
        Obstacle avoidance loss.

        Args:
            - x_batch: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - obstacle avoidance loss of shape (1, 1).
        """
        x_inds = torch.LongTensor(range(0, x_batch.shape[-2], 4)).to(x_batch.device)    # TODO
        y_inds = torch.LongTensor(range(1, x_batch.shape[-2], 4)).to(x_batch.device)    # TODO
        qx = torch.index_select(x_batch, dim=-2, index=x_inds)   # x of all agents. shape = (*batch_dim, T, n_agents, 1)
        qy = torch.index_select(x_batch, dim=-2, index=y_inds)   # y of all agents. shape = (*batch_dim, T, n_agents, 1)
        # batch over all samples and all times of [x agent 1, y agent 1, ..., x agent n, y agent n]
        q = torch.cat((qx,qy), dim=-1).view(*x_batch.shape[0:-2], 1, -1).squeeze(dim=-2)    # shape = (*batch_dim, T, 2*n_agents)
        # sum up loss due to each obstacle #TODO
        for ind, (center, cov) in enumerate(zip(self.obstacle_centers, self.obstacle_covs)):
            if ind == 0:
                loss_obst = normpdf(q, mu=center, cov=cov)   # shape = (*batch_dim, T)
            else:
                loss_obst += normpdf(q, mu=center, cov=cov)  # shape = (*batch_dim, T)
        # average over time steps
        loss_obst = loss_obst.sum(-1) / loss_obst.shape[-1]    # shape = (*batch_dim)
        return loss_obst.reshape(*x_batch.shape[0:-3], 1, 1)

    def f_loss_ca(self, x_batch):
        """
        Collision avoidance loss.

        Args:
            - x_batch: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.


        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = self.min_dist + 0.2
        # compute pairwise distances
        distance_sq = get_pairwise_distance_sq(x_batch, self.n_agents)              # shape = (*batch_dim, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * self.mask).sum((-1, -2))/2        # shape = (*batch_dim, T)
        # average over time steps
        loss_ca = loss_ca.sum(-1)/loss_ca.shape[-1]
        # reshape to *batch_dim, 1, 1
        loss_ca = loss_ca.reshape(*x_batch.shape[0:-3], 1, 1)
        return loss_ca

    def count_collisions(self, x_batch, return_col_mat=False):
        """
        Count the number of collisions between agents.

        Args:
            - x_batch: tensor of shape (*batch_dim, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - number of collisions between agents.
        """
        return count_collisions(x_batch, n_agents=self.n_agents, min_dist=self.min_dist, return_col_mat=return_col_mat)

def count_collisions(x_batch, n_agents, min_dist, return_col_mat=False):
    """
    Count the number of collisions between agents.

    Args:
        - x_batch: tensor of shape (*batch_dim, T, state_dim, 1)
            concatenated states of all agents on the third dimension.

    Return:
        - number of collisions between agents.
    """
    if len(x_batch.shape) == 3:
        x_batch = x_batch.reshape(*x_batch.shape, 1)
    distance_sq = get_pairwise_distance_sq(x_batch, n_agents)  # shape = (*batch_dim, T, n_agents, n_agents)
    col_matrix = (0.0001 < distance_sq) * (distance_sq < min_dist ** 2)  # Boolean collision matrix of shape (*batch_dim, T, n_agents, n_agents)
    n_coll = col_matrix.sum().item()    # all collisions at all times and across all rollouts
    if not return_col_mat:
        return n_coll/2                     # each collision is counted twice
    else:
        return n_coll/2, col_matrix

def get_pairwise_distance_sq(x_batch, n_agents):
    """
    Squared distance between pairwise agents.

    Args:
        - x_batch: tensor of shape (*batch_dim, T, state_dim, 1)
            concatenated states of all agents on the third dimension.

    Return:
        - matrix of shape (*batch_dim, T, n_agents, n_agents) of squared pairwise distances.
    """
    # collision avoidance:
    x_inds = torch.LongTensor(range(0, x_batch.shape[-2], 4))
    y_inds = torch.LongTensor(range(1, x_batch.shape[-2], 4))
    x_inds = torch.LongTensor(range(0, x_batch.shape[-2], 4)).to(x_batch.device)    # TODO
    y_inds = torch.LongTensor(range(1, x_batch.shape[-2], 4)).to(x_batch.device)    # TODO
    x_agents = torch.index_select(x_batch, dim=-2, index=x_inds)   # x of all agents. shape = (*batch_dim, T, n_agents, 1)
    y_agents = torch.index_select(x_batch, dim=-2, index=y_inds)   # y of all agents. shape = (*batch_dim, T, n_agents, 1)
    shape = [1,]*(len(x_agents.shape) - 1) + [n_agents,]
    deltaqx = x_agents.repeat(shape) - x_agents.repeat(shape).transpose(-2, -1)   # shape = (*batch_dim, T, n_agents, n_agents)
    deltaqy = y_agents.repeat(shape) - y_agents.repeat(shape).transpose(-2, -1)   # shape = (*batch_dim, T, n_agents, n_agents)
    distance_sq = deltaqx ** 2 + deltaqy ** 2             # shape = (*batch_dim, T, n_agents, n_agents)
    return distance_sq


def normpdf(q, mu, cov):  #TODO
    """
    PDF of normal distribution with mean "mu" and covariance "cov".

    Args:
        - q: shape(S, T, state_dim_per_agent)
        - mu:
        - cov:

    Return:
            -
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2, dim=-1)
    for ind, qi in enumerate(qs):
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        nom = torch.exp((-0.5 * (qi - mu)**2 / cov).sum(-1))
        # out = out + num/den
        if ind == 0:
            out = nom/den
        else:
            out += nom/den
    return out
