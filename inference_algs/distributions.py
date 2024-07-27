import torch, sys, os, time, math, copy
from collections import OrderedDict
from torch.func import stack_module_state, functional_call
from pyro.distributions import Normal, Uniform
from torch.distributions import Distribution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from loss_functions import *
from plants import CLSystem
from controllers.abstract import NNController, AffineController
from utils.assistive_functions import to_tensor, WrapLogger
from controllers import PerfBoostController

class GibbsPosterior():

    def __init__(
        self, loss_fn, lambda_, prior_dict,
        # attributes of the CL system
        controller, sys,
        # misc
        logger=None, num_ensemble_models=1
    ):
        self.lambda_ = to_tensor(lambda_)
        self.loss_fn = loss_fn
        self.logger = WrapLogger(logger)

        # Controller params will be set during training and the resulting CL system is use for evaluation.
        self.generic_cl_system = CLSystem(sys, controller, random_seed=None)

        # set prior
        self._set_prior(prior_dict)

        # init ensemble models
        self.num_ensemble_models = num_ensemble_models
        self.ensemble_models = [CLSystem(copy.deepcopy(sys), copy.deepcopy(controller), random_seed=None) for _ in range(self.num_ensemble_models)]
        # Construct a "stateless" version of one of the models: parameters are meta Tensors and do not have storage.
        self.ensemble_base_model = CLSystem(
            copy.deepcopy(sys), copy.deepcopy(controller), random_seed=None
        ).to('meta')

    def _log_prob_likelihood(self, params, train_data):
        """
        Args:
            params: of shape (param_batch_size, num_controller_params)
        """
        L = params.shape[0]

        param_ind = 0
        loss_val = None
        for _ in range(math.ceil(L/self.num_ensemble_models)):
            model_ind = 0

            # set params to controllers
            for _ in range(self.num_ensemble_models):
                if param_ind==L:
                    break
                self.ensemble_models[model_ind].controller.set_parameters_as_vector(
                    params[param_ind, :].reshape(1,-1)
                )
                model_ind += 1
                param_ind += 1
            used_ensemble_models = self.ensemble_models[0:model_ind]

            # stack all ensemble models
            ensemble_params_mdl, ensemble_buffers_mdl = stack_module_state(used_ensemble_models)

            # rollout
            xs, us = torch.vmap(
                self.ensemble_functional_model,
                in_dims=(0, 0, None)
            )(ensemble_params_mdl, ensemble_buffers_mdl, train_data)

            # compute loss
            if isinstance(self.loss_fn, RobotsLoss) or isinstance(self.loss_fn, LQLossFH):
                for ind in range(xs.shape[0]):
                    loss_val_tmp = self.loss_fn.forward(xs[ind, :, :, :], us[ind, :, :, :])
                    if loss_val is None:
                        loss_val = [loss_val_tmp]
                    else:
                        loss_val.append(loss_val_tmp)
            elif isinstance(self.loss_fn, RobotsLossMultiBatch) or isinstance(self.loss_fn, LQLossFHMultiBatch):
                loss_val_tmp = self.loss_fn.forward(xs, us)
                if loss_val is None:
                    loss_val = [loss_val_tmp]
                else:
                    loss_val.append(loss_val_tmp)
            else:
                raise NotImplementedError

        loss_val = torch.cat(loss_val)

        assert param_ind==L
        assert loss_val.shape[0]==L and loss_val.shape[1]==1, loss_val.shape

        return loss_val

    def log_prob(self, params, train_data):
        '''
        params is of shape (L, -1)
        '''
        assert len(params.shape)<3
        if len(params.shape)==1:
            params = params.reshape(1, -1)
        L = params.shape[0]
        assert params.grad_fn is not None
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
        if isinstance(self.generic_cl_system.controller, PerfBoostController):
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
        if isinstance(self.generic_cl_system.controller, NNController):
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
        elif isinstance(self.generic_cl_system.controller, AffineController):
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


    def ensemble_functional_model(self, params, buffers, x):
        return functional_call(self.ensemble_base_model, (params, buffers), (x,))


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
