import pickle
import torch, os, sys, time, copy
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from utils.plot_functions import plot_trajectories
from inference_algs.normflow_assist import eval_norm_flow

def fit_norm_flow(
    nfm, sys, ctl_generic, logger, loss_fn, save_folder, 
    train_data_full, test_data, plot_data,
    optimizer, epochs, log_epoch, annealing, anneal_iter, return_best, 
    early_stopping, n_logs_no_change, tol_percentage,
    validation_frac=0.25, num_samples_nf_train=100, num_samples_nf_eval=100,
):
    col_av = True
    t_ext = plot_data.shape[1]
    # validation data
    if early_stopping:
        assert validation_frac>0
        valid_imp_queue = [100]*n_logs_no_change   # don't stop at the beginning
    if return_best:
        assert validation_frac>0 
        best_loss = 1e10
        best_model_dict = None
    if validation_frac>0:
        valid_inds = torch.randperm(train_data_full.shape[0])[:int(validation_frac*train_data_full.shape[0])]
        train_inds = [ind for ind in range(train_data_full.shape[0]) if ind not in valid_inds]
        valid_data = train_data_full[valid_inds, :, :] if len(valid_inds)>0 else None
        train_data = train_data_full[train_inds, :, :]
    else:
        train_data = train_data_full
        valid_data = None
    if valid_data is None:
        return_best = False

    # ------------ Test initial model ------------
    # plot closed-loop trajectories by sampling controller from untrained nfm and base distribution
    with torch.no_grad():
        for dist, dist_name in zip([nfm.q0, nfm], ['base', 'init']):
            logger.info('Plotting closed-loop trajectories for ' + dist_name + ' flow model.')
            if dist_name=='base':
                z = dist.sample(num_samples_nf_eval)
                z_mean = dist.loc
            else:
                z, _ = dist.sample(num_samples_nf_eval)
                z_mean = torch.mean(z, axis=0)
            _, xs_z_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z
            )
            _, xs_z_mean_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z_mean
            )
            plot_trajectories(
                torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
                sys.xbar, sys.n_agents, text='CL - ' + dist_name + ' flow', T=t_ext,
                save_folder=save_folder, filename='CL_'+dist_name+'.pdf',
                obstacle_centers=loss_fn.obstacle_centers,
                obstacle_covs=loss_fn.obstacle_covs,
                plot_collisions=True, min_dist=loss_fn.min_dist
            )

    # evaluate on the train data
    logger.info('\n[INFO] evaluating the base distribution on %i training rollouts.' % train_data.shape[0])
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm.q0, sys=sys, ctl_generic=ctl_generic, data=train_data,
        num_samples=num_samples_nf_eval, loss_fn=loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % train_loss
    if col_av:
        msg += ' -- Average number of collisions = %i' % train_num_col
    logger.info(msg)
    # evaluate on the train data
    logger.info('\n[INFO] evaluating the initial flow on %i training rollouts.' % train_data.shape[0])
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
        num_samples=num_samples_nf_eval, loss_fn=loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % train_loss
    if col_av:
        msg += ' -- Average number of collisions = %i' % train_num_col
    logger.info(msg)

    # ------------ Train NormFlows ------------
    nf_loss_hist = [None]*(1+epochs)
    t = time.time()
    epoch = 0
    stopped = False
    while not stopped:
        optimizer.zero_grad()
        # loss 
        if annealing:
            nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + epoch / anneal_iter]))
        else:
            nf_loss = nfm.reverse_kld(num_samples_nf_train)
        nf_loss_hist[epoch]= nf_loss.to('cpu').data.numpy()
        # take a step
        nf_loss.backward()
        optimizer.step()

        # print info
        if epoch%log_epoch == 0:
            with torch.no_grad():
                # evaluate some sampled controllers on train data
                z, _ = nfm.sample(num_samples_nf_eval)
                loss_z_train, _ = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=train_data,
                    loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z
                )
                # evaluate mean of sampled controllers on train data
                z_mean = torch.mean(z, axis=0).reshape(1, -1)
                loss_z_mean_train, _ = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=train_data,
                    loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z_mean
                )
                # evaluate some sampled controllers on valid data
                if not valid_data is None:
                    loss_z_valid, _ = eval_norm_flow(
                        sys=sys, ctl_generic=ctl_generic, data=valid_data,
                        loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z
                    )
                    # evaluate mean of sampled controllers on valid data
                    loss_z_mean_valid, _ = eval_norm_flow(
                        sys=sys, ctl_generic=ctl_generic, data=valid_data,
                        loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z_mean
                    )
            # log nf loss 
            msg = 'Epoch: %i --- NF loss: %.2f'% (epoch, nf_loss_hist[epoch])
            msg += ' ---||--- elapsed time: %.0f s' % (time.time() - t)
            msg += ' --- train loss %f' % loss_z_train + ' --- train loss of mean %f' % loss_z_mean_train
            if not valid_data is None:
                msg += ' --- valid loss %f' % loss_z_valid + ' --- valid loss of mean %f' % loss_z_mean_valid
            
                # compare with the best valid loss
                imp = 100 * (best_loss-loss_z_valid)/best_loss
                # update best model
                if return_best and loss_z_valid < best_loss:
                    best_loss = loss_z_valid
                    best_model_dict = copy.deepcopy(nfm.state_dict())
                    msg += ' --- best model updated'
            
            # early stopping
            if early_stopping and not valid_data is None:
                # add the current valid loss to the queue
                valid_imp_queue.pop(0)
                valid_imp_queue.append(imp)
                # check if there is no improvement
                if all([valid_imp_queue[i] <tol_percentage for i in range(n_logs_no_change)]):
                    msg += ' ---||--- early stopping at epoch %i' % (epoch)
                    logger.info(msg)
                    stopped = True
            
            logger.info(msg)

            # plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(nf_loss_hist, label='loss')
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'loss.pdf'))
            
            # plot closed_loop
            _, xs_z_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z
            )
            _, xs_z_mean_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z_mean
            )
            plot_trajectories(
                torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
                sys.xbar, sys.n_agents, text='CL - itr '+str(epoch), T=t_ext,
                save_folder=save_folder, filename='CL_itr '+str(epoch+1)+'.pdf',
                obstacle_centers=loss_fn.obstacle_centers,
                obstacle_covs=loss_fn.obstacle_covs,
                plot_collisions=True, min_dist=loss_fn.min_dist
            )

        # next epoch
        epoch += 1
        if epoch == epochs+1:
            stopped = True

    # save nf model
    if return_best:
        nfm.load_state_dict(best_model_dict)
    torch.save(nfm.state_dict(), os.path.join(save_folder, 'trained_nfm'))
        
    # ------ Evaluate the trained model ------
    # evaluate on the train data
    logger.info('\n[INFO] evaluating the trained flow on the entire %i training rollouts.' % train_data_full.shape[0])
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data_full,
        num_samples=num_samples_nf_eval, loss_fn=loss_fn, count_collisions=col_av, return_av=False
    )
    train_loss_av = sum(train_loss)/len(train_loss)
    train_num_col_av = sum(train_num_col)/len(train_num_col)
    msg = 'Average loss: %.4f' % train_loss_av
    if col_av:
        msg += ' -- total number of collisions = %i' % train_num_col_av
    logger.info(msg)

    # evaluate on the test data
    logger.info('\n[INFO] evaluating the trained flow on %i test rollouts.' % test_data.shape[0])
    test_loss, test_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=test_data,
        num_samples=num_samples_nf_eval, loss_fn=loss_fn, count_collisions=col_av, return_av=False
    )
    test_loss_av = sum(test_loss)/len(test_loss)
    test_num_col_av = sum(test_num_col)/len(test_num_col)
    msg = 'Average loss: %.4f' % test_loss_av
    if col_av:
        msg += ' -- total number of collisions = %i' % test_num_col_av
    logger.info(msg)

    res_dict = {
        'train_loss':train_loss, 'train_num_col':train_num_col,
        'train_loss_av': train_loss_av, 'train_num_col_av': train_num_col_av,
        'test_loss':test_loss, 'test_num_col':test_num_col,
        'test_loss_av': test_loss_av, 'test_num_col_av': test_num_col_av
    }
    with open(os.path.join(save_folder, 'res_dict.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)

    # plot closed-loop trajectories using the trained controller
    logger.info('Plotting closed-loop trajectories using the trained controller...')
    with torch.no_grad():
        z, _ = nfm.sample(100)
        z_mean = torch.mean(z, axis=0)
        _, xs_z_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z
        )
        _, xs_z_mean_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=loss_fn, count_collisions=False, return_traj=True, params=z_mean
        )
        plot_trajectories(
            torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
            sys.xbar, sys.n_agents, text='CL - trained flow', T=t_ext,
            save_folder=save_folder, filename='CL_trained.pdf',
            obstacle_centers=loss_fn.obstacle_centers,
            obstacle_covs=loss_fn.obstacle_covs,
            plot_collisions=True, min_dist=loss_fn.min_dist
        )

    return res_dict, os.path.join(save_folder, 'trained_nfm')