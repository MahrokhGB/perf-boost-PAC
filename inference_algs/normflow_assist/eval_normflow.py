import torch

def eval_norm_flow(sys, ctl_generic, data, loss_fn, count_collisions, return_traj=False, num_samples=None, nfm=None, params=None):
    '''
    evaluate normalizing flow model.

    - params: if None, sample from nfm. o.w., evaluate the given params.
    - nfm: the normflow model used to sample params if params is None.
    '''
    with torch.no_grad():
        # sample from nfm. params is of shape(num_samples, dim_param)
        if params is None:
            assert not num_samples is None, '[ERR] number of parameters sampled from the nfm model to evaluate it must be specified.'
            assert not nfm is None, '[ERR] the normalizing flow model (nfm) must be given to sample from.'
            if nfm.__class__.__name__=='NormalizingFlow':
                params, _ = nfm.sample(num_samples)
            else:
                params = nfm.sample(num_samples)
        else:
            if not nfm is None:
                print('[WARN] nfm model was provided but not used.')
            if not num_samples is None:
                print('[WARN] num_samples was provided but not used.')
            num_samples = params.shape[0]

        # repeat data if num_samples>1
        if params.ndim==1:
            params = params.reshape(1, -1)
        elif params.ndim==2 and params.shape[0]>1:
            data = data.unsqueeze(1).expand(-1, params.shape[0], -1, -1)

        # set params to controller
        ctl_generic.set_parameters_as_vector(params)

        # rollout. xs is of shape (data_batch_size, num_samples, T, dim_states)
        xs, _, us = sys.rollout(
            controller=ctl_generic,
            data=data,
        )

        # compute loss. loss_val is of shape (num_samples, )
        loss_val = loss_fn.forward(xs.transpose(0,1), us.transpose(0,1))
        if loss_val.ndim>2:
            loss_val = loss_val.squeeze(-1, -2)
        elif loss_val.ndim>1:
            loss_val = loss_val.squeeze(-1)

        # count collisions
        if count_collisions:
            num_col = [None]*num_samples
            for param_ind in range(num_samples):
                num_col[param_ind] = loss_fn.count_collisions(xs[:, param_ind, :, :])

        if count_collisions:
            if not return_traj:
                return sum(loss_val)/len(loss_val), sum(num_col)/len(num_col)
            else:
                return sum(loss_val)/len(loss_val), sum(num_col)/len(num_col), xs
        else:
            if not return_traj:
                return sum(loss_val)/len(loss_val)
            else:
                return sum(loss_val)/len(loss_val), xs
