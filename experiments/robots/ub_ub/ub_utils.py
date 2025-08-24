import sys, os, torch, math
import normflows as nf
from scipy.optimize import fsolve, least_squares

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from inference_algs.normflow_assist.mynf import NormalizingFlow
from inference_algs.normflow_assist import eval_norm_flow

def get_neg_log_zhat_over_lambda(
    sys, ctl_generic, train_data, bounded_loss_fn, lambda_, prior, num_prior_samples,
    return_stats=True, batch_size=1000
):
    '''

    '''
    sample_counter = 0
    Z_hat_norm = 0
    normalization_factor = None
    if return_stats:
        stats={'mean':0, 'min':1e6}
    while sample_counter<num_prior_samples:
        samples_in_batch = min(batch_size, num_prior_samples-sample_counter)
        if isinstance(prior, NormalizingFlow) or isinstance(prior, nf.NormalizingFlow):
            prior_samples, _ = prior.sample(samples_in_batch)
        else:
            prior_samples = prior.sample(torch.Size([samples_in_batch]))
        # evaluate samples controllers
        ctl_generic.reset()
        ctl_generic.emme.hard_reset()
        train_loss_batch, _ = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=train_data,
            num_samples=None,nfm=None,
            params=prior_samples,
            loss_fn=bounded_loss_fn,
            count_collisions=True, return_av=False
        )
        # weighting factors
        factor_old = sample_counter/(sample_counter+samples_in_batch)
        factor_new = 1/(sample_counter+samples_in_batch)
        # compute stats
        if return_stats:
            stats={
                'mean':stats['mean']*factor_old+torch.mean(train_loss_batch)*factor_new,
                'min':torch.min(stats['min'], torch.min(train_loss_batch))
            }
        # compute e^{- lambda loss}
        if normalization_factor is None:
            normalization_factor =  torch.min(train_loss_batch)
            exp_train_loss_batch = torch.exp(-lambda_*(train_loss_batch-normalization_factor))
        # update the average e^{- lambda train loss}
        Z_hat_norm = Z_hat_norm*factor_old + torch.sum(exp_train_loss_batch)*factor_new
        # increase counter
        sample_counter += samples_in_batch
        print('computed for '+str(sample_counter)+' prior samples.')
    assert sample_counter==num_prior_samples

    # compute -1/lambda ln(Zhat)
    log_Z_hat = math.log(Z_hat_norm) - lambda_*normalization_factor

    if not return_stats:
        return - 1/lambda_*log_Z_hat
    else:
        return - 1/lambda_*log_Z_hat, stats





def get_min_np(thresh, delta, lambda_, init_condition=10000, loss_bound=1, constrained=True, max_tries=20):
    '''
    find n_p to have epsilon/lambda approx equal to thresh
    '''
    return get_relation(thresh=thresh, delta=delta, n_p=None, lambda_=lambda_, init_condition=init_condition, loss_bound=loss_bound, constrained=constrained, max_tries=max_tries)

def get_max_lambda(thresh, delta, n_p, init_condition=10000, loss_bound=1, constrained=True, max_tries=20):
    '''
    find lambda to have epsilon/lambda approx equal to thresh
    '''
    return get_relation(thresh=thresh, delta=delta, n_p=n_p, lambda_=None, init_condition=init_condition, loss_bound=loss_bound, constrained=constrained, max_tries=max_tries)

def get_relation(thresh, delta, n_p=None, lambda_=None, init_condition=10000, loss_bound=1, constrained=True, max_tries=20):
    '''
    find n_p to have epsilon/lambda approx equal to thresh
    or
    find lambda to have epsilon/lambda approx equal to thresh
    '''
    assert (lambda_ is None or n_p is None) and not (lambda_ is None and n_p is None)
    num_tries = 0
    accepted = False
    while not accepted:
        # define function to solve
        if lambda_ is None:
            func = lambda x: thresh-get_epsilon(num_prior_samples=n_p, delta=delta, lambda_=x, loss_bound=loss_bound)/x
            min_bound = 0
        else:
            func = lambda x: thresh-get_epsilon(num_prior_samples=x, delta=delta, lambda_=lambda_, loss_bound=loss_bound)/lambda_
            min_bound = 1

        # solve with current initial guess
        if not constrained:
            root = fsolve(func, x0=[init_condition])
            root = max(min_bound, root[0])
        else:
            root = least_squares(func, x0=[init_condition], bounds = ((min_bound), (math.inf)))
            root = max(min_bound, root.x[0])
        # check error
        error = func(root)
        if abs(error) <= thresh/10:
            accepted=True
        else:
            if num_tries>max_tries:
                print('[Err] Could not find the solution for thresh = '+str(thresh))
                exit()
            else:
                num_tries += 1
            # update initial guess
            if (error < 0 and n_p is None) or (error>0 and lambda_ is None):
                init_condition = init_condition**1.05 if not init_condition==1 else init_condition*2
                # init_condition = init_condition*2**10
                print('Increased the initial guess')
                if init_condition > 2**500:
                    print('[Err] Could not find the solution for thresh = '+str(thresh))
                    exit()
            else:
                init_condition = init_condition**0.95 if not init_condition==1 else init_condition/2
                # init_condition = init_condition/2**10
                print('Decreased the initial guess')
                if init_condition < 2:
                    print('[Err] Could not find the solution for thresh = '+str(thresh))
                    exit()

    if n_p is None:
        root = round(root)

    return root

def get_epsilon(num_prior_samples, delta, lambda_, loss_bound=1):
    assert num_prior_samples>=1, num_prior_samples
    term1 = (num_prior_samples/2*math.log(1/delta))**0.5
    try:
        exp_lambda_c = math.exp(lambda_*loss_bound)
        return term1 * math.log(1+(exp_lambda_c-1)/num_prior_samples)
    except:
        # first-order Taylor approximation: term1 * (exp_lambda_c-1)/num_prior_samples
        # if exp_lambda_c is too large, can further approximate as 
        # term1 * exp_lambda_c / num_prior_samples, or equivalently, as follows:
        # only valid if (exp_lambda_c-1)/num_prior_samples is very small.
        print('\n[INFO] compute epsilon in the upper bound using first-order Taylor expansion.')
        return term1 * math.exp(lambda_*loss_bound - math.log(num_prior_samples))


def get_mcdim_ub(
    sys, ctl_generic, train_data, bounded_loss_fn, num_prior_samples, delta, lambda_, C, prior=None,
    deltahat=None, batch_size=1000, return_keys=['neg_log_zhat_over_lambda', 'epsilon/lambda_', 'ub_const']
):
    deltahat = delta if deltahat is None else deltahat
    num_rollouts = train_data.shape[0]
    assert num_rollouts>0, num_rollouts

    n_p_min = math.ceil(
        (1-math.exp(-lambda_*C)/lambda_/C)**2 * math.log(1/deltahat) / 2
    )
    assert num_prior_samples>n_p_min

    # epsilon
    if 'epsilon/lambda_' in return_keys:
        epsilon = get_epsilon(
            num_prior_samples=num_prior_samples, delta=delta, lambda_=lambda_, loss_bound=C
        )
    else:
        epsilon = torch.Tensor([0])

    # constant term
    if 'ub_const' in return_keys:
        ub_const = 1/lambda_*math.log(1/delta) + lambda_*C**2/8/num_rollouts
    else:
        ub_const = torch.Tensor([0])

    # Zhat
    if 'neg_log_zhat_over_lambda' in return_keys:
        assert not prior is None
        neg_log_zhat_over_lambda = get_neg_log_zhat_over_lambda(
            sys=sys, ctl_generic=ctl_generic, train_data=train_data,
            bounded_loss_fn=bounded_loss_fn, prior=prior,
            num_prior_samples=1000, #num_prior_samples, # TODO
            lambda_=lambda_, return_stats=False,
            batch_size=batch_size
        )
    else:
        neg_log_zhat_over_lambda = torch.Tensor([0])

    neg_log_zhat_over_lambda = neg_log_zhat_over_lambda.item()
    mcdim_ub = {
        'tot':neg_log_zhat_over_lambda + epsilon/lambda_ + ub_const,
        'neg_log_zhat_over_lambda':neg_log_zhat_over_lambda,
        'epsilon/lambda_':epsilon/lambda_,
        'ub_const':ub_const
    }

    return mcdim_ub