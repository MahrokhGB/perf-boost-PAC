import torch, sys, os, math, copy
from collections import OrderedDict
from torch.func import stack_module_state, functional_call
from pyro.distributions import Normal, Uniform
from torch.distributions import Distribution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from loss_functions import *
from plants import CLSystem
from controllers import NNController, AffineController, PerfBoostController
from utils.assistive_functions import to_tensor, WrapLogger

class GibbsPosterior():

    def __init__(
        self, loss_fn, lambda_, prior_dict,
        # attributes of the CL system
        controller, sys,
        # misc
        logger=None
    ):
        self.lambda_ = to_tensor(lambda_)
        self.loss_fn = loss_fn
        self.logger = WrapLogger(logger)

        # Controller params will be set during training and the resulting CL system is use for evaluation.
        self.generic_cl_system = CLSystem(sys, controller, random_seed=None)

        # set prior
        self._set_prior(prior_dict)

    def _log_prob_likelihood(self, params, train_data):
        if params.ndim==1:
            params = params.reshape(1, -1)
        elif params.ndim==2 and params.shape[0]>1:
            train_data = train_data.unsqueeze(1).expand(-1, params.shape[0], -1, -1)

        # set params to controller
        cl_system = self.generic_cl_system
        cl_system.controller.set_parameters_as_vector(
            params
        )
        # rollout
        xs, _, us = cl_system.rollout(train_data)
        # compute loss
        loss_val = self.loss_fn.forward(xs.transpose(0,1), us.transpose(0,1))

        return loss_val

    def log_prob(self, params, train_data):
        '''
        params is of shape (L, -1)
        '''
        # assert len(params.shape)<3
        if len(params.shape)==1:
            params = params.reshape(1, -1)
        L = params.shape[0]
        # assert params.grad_fn is not None
        lpl = self._log_prob_likelihood(params, train_data)
        lpl = lpl.reshape(L) # TODO
        lpp = self._log_prob_prior(params)
        lpp = lpp.reshape(L)
        # assert not (lpl.grad_fn is None or lpp.grad_fn is None)
        '''
        NOTE: To debug, remove the effect of the prior by returning -lpl
        '''
        # return -lpl
        return lpp - self.lambda_ * lpl

    def sample_params_from_prior(self, shape):
        # shape is torch.Size()
        return self.prior.sample(shape)

    def _log_prob_prior(self, params):
        return self.prior.log_prob(params)

    def _param_dist(self, name, dist):
        assert type(name) == str
        assert isinstance(dist, torch.distributions.Distribution)
        if isinstance(dist.base_dist, Normal):
            dist.base_dist.loc = dist.base_dist.loc.to(device)
            dist.base_dist.scale = dist.base_dist.scale.to(device)
        elif isinstance(dist.base_dist, Uniform):
            dist.base_dist.low = dist.base_dist.low.to(device)
            dist.base_dist.high = dist.base_dist.high.to(device)
        if name in list(self._param_dists.keys()):
            self.logger.info('[WARNING] name ' + name + 'was already in param dists')
        # assert name not in list(self._param_dists.keys())
        assert hasattr(dist, 'rsample')
        self._param_dists[name] = dist

        return dist

    def parameter_shapes(self):
        param_shapes_dict = OrderedDict()
        for name, dist in self._param_dists.items():
            param_shapes_dict[name] = dist.event_shape
        return param_shapes_dict

    def get_forward_cl_system(self, params):
        cl_system = self.generic_cl_system
        cl_system.controller.set_parameters_as_vector(params)
        return cl_system

    def _set_prior(self, prior_dict):
        self._params = OrderedDict()
        self._param_dists = OrderedDict()
        # ------- set prior -------
        # set prior for REN controller
        if self.generic_cl_system.controller.__class__.__name__=='PerfBoostController':
            for name, shape in self.generic_cl_system.controller.get_parameter_shapes().items():
                nelement = torch.empty(*shape).nelement()   # number of total elements in the tensor
                # Gaussian prior
                if prior_dict['type'] == 'Gaussian':
                    if not (name+'_loc' in prior_dict.keys() or name+'_scale' in prior_dict.keys()):
                        self.logger.info('[WARNING]: prior for ' + name + ' was not provided. Replaced by default.')
                    dist = Normal(
                        loc=prior_dict.get(name+'_loc', 0)*torch.ones(nelement, device=device),
                        scale=prior_dict.get(name+'_scale', 1)*torch.ones(nelement, device=device)
                    )
                # Uniform prior
                elif prior_dict['type'] == 'Uniform':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                # set dist
                self._param_dist(name, dist.to_event(1))
        # set prior for NN controller
        elif self.generic_cl_system.controller.__class__.__name__ == 'NNController':
            # check if prior is provided
            for name in ['weight', 'bias']:
                if not name+'_loc' in prior_dict.keys():
                    if prior_dict['type'] == 'Gaussian':
                        self.logger.info('[WARNING]: prior loc for ' + name + ' was not provided. Replaced by 0.')
                        prior_dict[name+'_loc'] = 0
                    else:
                        raise NotImplementedError
                if not name+'_scale' in prior_dict.keys():
                    if prior_dict['type'] == 'Gaussian':
                        self.logger.info('[WARNING]: prior scale for ' + name + ' was not provided. Replaced by 1.')
                        prior_dict[name+'_scale'] = 1
                    else:
                        raise NotImplementedError
            # set prior for hidden layers
            for i in range(1, self.generic_cl_system.controller.n_layers + 1):
                # Gaussian prior
                for name in ['weight', 'bias']:
                    param = getattr(
                        getattr(self.generic_cl_system.controller, 'fc_%i'%(i)),
                        name
                    )
                    if prior_dict['type'] == 'Gaussian':
                        dist = Normal(
                            loc=prior_dict.get(name+'_loc', 0)*torch.ones(param.nelement(), device=device),
                            scale=prior_dict.get(name+'_scale', 1)*torch.ones(param.nelement(), device=device)
                        )
                    else:
                        raise NotImplementedError
                    # set dist
                    self._param_dist('fc_%i'%(i)+'.'+name, dist.to_event(1))
            # set prior for output layers
            for name in ['weight', 'bias']:
                param = getattr(
                    getattr(self.generic_cl_system.controller, 'out'),
                    name
                )
                if prior_dict['type'] == 'Gaussian':
                    dist = Normal(
                        loc=prior_dict.get(name+'_loc', 0)*torch.ones(param.nelement(), device=device),
                        scale=prior_dict.get(name+'_scale', 1)*torch.ones(param.nelement(), device=device)
                    )
                else:
                    raise NotImplementedError
                # set dist
                self._param_dist('out.'+name, dist.to_event(1))
        elif self.generic_cl_system.controller.__class__.__name__ == 'AffineController':
            for name, shape in self.generic_cl_system.controller.parameter_shapes().items():
                # Gaussian prior
                if prior_dict['type_'+name[0]].startswith('Gaussian'):
                    if not (name+'_loc' in prior_dict.keys() or name+'_scale' in prior_dict.keys()):
                        self.logger.info('[WARNING]: prior for ' + name + ' was not provided. Replaced by default.')
                    dist = Normal(
                        loc=prior_dict.get(name+'_loc', 0)*torch.ones(shape).flatten().to(device),
                        scale=prior_dict.get(name+'_scale', 1)*torch.ones(shape).flatten().to(device)
                    )
                # Uniform prior
                elif prior_dict['type_'+name[0]] == 'Uniform':
                    assert (name+'_low' in prior_dict.keys()) and (name+'_high' in prior_dict.keys())
                    dist = Uniform(
                        low=prior_dict[name+'_low']*torch.ones(shape).flatten().to(device),
                        high=prior_dict[name+'_high']*torch.ones(shape).flatten().to(device)
                    )
                else:
                    raise NotImplementedError
                # set dist
                self._param_dist(name, dist.to_event(1))
        else:
            raise NotImplementedError

        # check that parameters in prior and controller are aligned
        if not isinstance(self.generic_cl_system.controller, NNController):
            for param_name_cont, param_name_prior in zip(self.generic_cl_system.controller.get_named_parameters().keys(), self._param_dists.keys()):
                assert param_name_cont == param_name_prior, param_name_cont + 'in controller did not match ' + param_name_prior + ' in prior'

        self.prior = CatDist(self._param_dists.values())


# -------------------------
# -------------------------


# -------------------------
# -------------------------

class CatDist(Distribution):

    def __init__(self, dists, reduce_event_dim=True):
        assert all([len(dist.event_shape) == 1 for dist in dists])
        assert all([len(dist.batch_shape) == 0 for dist in dists])
        self.reduce_event_dim = reduce_event_dim
        self.dists = dists
        self._event_shape = torch.Size((sum([dist.event_shape[0] for dist in self.dists]),))

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='sample')

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='rsample')

    def log_prob(self, value):
        idx = 0
        log_probs = []
        for dist in self.dists:
            n = dist.event_shape[0]
            if value.ndim == 1:
                val = value[idx:idx+n]
            elif value.ndim == 2:
                val = value[:, idx:idx + n]
            elif value.ndim == 2:
                val = value[:, :, idx:idx + n]
            else:
                raise NotImplementedError('Can only handle values up to 3 dimensions')
            log_probs.append(dist.log_prob(val))
            idx += n

        for i in range(len(log_probs)):
            if log_probs[i].ndim == 0:
                log_probs[i] = log_probs[i].reshape((1,))

        if self.reduce_event_dim:
            return torch.sum(torch.stack(log_probs, dim=0), dim=0)
        return torch.stack(log_probs, dim=0)

    def mean(self):
        means = []
        for dist in self.dists:
            means.append(dist.mean.flatten())
        return torch.cat(means, dim=0)

    def stddev(self):
        stddevs = []
        for dist in self.dists:
            stddevs.append(dist.stddev.flatten())
        return torch.cat(stddevs, dim=0)

    def _sample(self, sample_shape, sample_fn='sample'):
        return torch.cat([getattr(d, sample_fn)(sample_shape).to(device) for d in self.dists], dim=-1)


class BlockwiseDist(Distribution):
    def __init__(self, priors):
        assert isinstance(priors, list)
        for prior in priors:
            assert isinstance(prior, Distribution)
        self.priors = priors
        self.num_priors = len(priors)

    def sample(self):
        res = torch.zeros(self.num_priors)
        for prior_num in range(self.num_priors):
            res[prior_num] = self.priors[prior_num].sample()
        return res
