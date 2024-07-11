import torch
import normflows as nf

def eval_norm_flow(nfm, sys, ctl_generic, data, num_samples, loss_fn, count_collisions=True):
    with torch.no_grad():
        if isinstance(nfm, nf.NormalizingFlow):
            z, _ = nfm.sample(num_samples)
        else:
            z = nfm.sample(num_samples)
        loss = [None]*num_samples
        num_col = [None]*num_samples
        for param_ind in range(num_samples):
            ctl_generic.set_parameters_as_vector(z[param_ind, :])
            x_log, _, u_log = sys.rollout(
                controller=ctl_generic,
                data=data, train=False,
            )
            # evaluate losses
            loss[param_ind] = loss_fn.forward(x_log, u_log)
            # count collisions
            if count_collisions:
                num_col[param_ind] = loss_fn.count_collisions(x_log)

    return sum(loss)/len(loss), sum(num_col)/len(num_col)