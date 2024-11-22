import torch, os, sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from utils.plot_functions import plot_trajectories
from inference_algs.normflow_assist import eval_norm_flow

def train_norm_flow(
    nfm, sys, ctl_generic, logger, bounded_loss_fn, save_folder, train_data, test_data, plot_data,
    optimizer, epochs, log_epoch, annealing, anneal_iter, num_samples_nf_train=100, num_samples_nf_eval=100,
):
    col_av = True
    num_train_rollouts = train_data.shape[0]
    t_ext = plot_data.shape[1]

    # ------------ Test initial model ------------
    # plot closed-loop trajectories by sampling controller from untrained nfm and base distribution
    with torch.no_grad():
        for dist, dist_name in zip([nfm.q0, nfm], ['base', 'init']):
            logger.info('Plotting closed-loop trajectories for ' + dist_name + ' flow model.')
            if dist_name=='base':
                z = dist.sample(num_samples_nf_eval)
            else:
                z, _ = dist.sample(num_samples_nf_eval)
            z_mean = torch.mean(z, axis=0)
            _, xs_z_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
            )
            _, xs_z_mean_plot = eval_norm_flow(
                sys=sys, ctl_generic=ctl_generic, data=plot_data,
                loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
            )
            plot_trajectories(
                torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
                sys.xbar, sys.n_agents, text='CL - ' + dist_name + ' flow', T=t_ext,
                save_folder=save_folder, filename='CL_'+dist_name+'.pdf',
                obstacle_centers=bounded_loss_fn.obstacle_centers,
                obstacle_covs=bounded_loss_fn.obstacle_covs,
                plot_collisions=True, min_dist=bounded_loss_fn.min_dist
            )
            # print(dist_name, ' mean ', z_mean[0:10])
            # print(dist_name, ' std ', torch.std(z, dim=0)[0:10])

    # evaluate on the train data
    logger.info('\n[INFO] evaluating the base distribution on %i training rollouts.' % num_train_rollouts)
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm.q0, sys=sys, ctl_generic=ctl_generic, data=train_data,
        num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % train_loss
    if col_av:
        msg += ' -- Average number of collisions = %i' % train_num_col
    logger.info(msg)
    # evaluate on the train data
    logger.info('\n[INFO] evaluating the initial flow on %i training rollouts.' % num_train_rollouts)
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
        num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % train_loss
    if col_av:
        msg += ' -- Average number of collisions = %i' % train_num_col
    logger.info(msg)

    # ------------ Train NormFlows ------------
    nf_loss_hist = [None]*epochs

    with tqdm(range(epochs)) as t:
        for it in t:
            optimizer.zero_grad()
            if annealing:
                nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + it / anneal_iter]))
            else:
                nf_loss = nfm.reverse_kld(num_samples_nf_train)
            nf_loss.backward()
            optimizer.step()

            nf_loss_hist[it] = nf_loss.to('cpu').data.numpy()

            # Eval and log
            if (it + 1) % log_epoch == 0 or it+1==epochs:
                with torch.no_grad():
                    # evaluate some sampled controllers
                    z, _ = nfm.sample(num_samples_nf_eval)
                    loss_z, xs_z = eval_norm_flow(
                        sys=sys, ctl_generic=ctl_generic, data=train_data,
                        loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
                    )
                    # evaluate mean of sampled controllers
                    z_mean = torch.mean(z, axis=0).reshape(1, -1)
                    print('mean ', z_mean[0,0:10])
                    print('std ', torch.std(z, axis=0)[0:10])
                    loss_z_mean, xs_z_mean = eval_norm_flow(
                        sys=sys, ctl_generic=ctl_generic, data=train_data,
                        loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
                    )

                # log nf loss
                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
                msg = 'Iter %i' % (it+1) + ' --- elapsed time: ' + elapsed_str  + ' --- norm flow loss: %f'  % nf_loss.item()
                msg += ' --- train loss %f' % loss_z + ' --- train loss of mean %f' % loss_z_mean
                logger.info(msg)

                # save nf model
                name = 'final' if it+1==epochs else 'itr '+str(it+1)
                if name == 'final':
                    torch.save(nfm.state_dict(), os.path.join(save_folder, name+'_nfm'))
                # plot loss
                plt.figure(figsize=(10, 10))
                plt.plot(nf_loss_hist, label='loss')
                plt.legend()
                plt.savefig(os.path.join(save_folder, 'loss.pdf'))
                plt.show()
                # plot closed_loop
                _, xs_z_plot = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=plot_data,
                    loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
                )
                _, xs_z_mean_plot = eval_norm_flow(
                    sys=sys, ctl_generic=ctl_generic, data=plot_data,
                    loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
                )
                plot_trajectories(
                    torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
                    sys.xbar, sys.n_agents, text="CL - "+name, T=t_ext,
                    save_folder=save_folder, filename='CL_'+name+'.pdf',
                    obstacle_centers=bounded_loss_fn.obstacle_centers,
                    obstacle_covs=bounded_loss_fn.obstacle_covs,
                    plot_collisions=True, min_dist=bounded_loss_fn.min_dist
                )

    # ------ Evaluate the trained model ------
    # evaluate on the train data
    logger.info('\n[INFO] evaluating the trained flow on %i training rollouts.' % num_train_rollouts)
    train_loss, train_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=train_data,
        num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % train_loss
    if col_av:
        msg += ' -- total number of collisions = %i' % train_num_col
    logger.info(msg)

    # evaluate on the test data
    logger.info('\n[INFO] evaluating the trained flow on %i test rollouts.' % test_data.shape[0])
    test_loss, test_num_col = eval_norm_flow(
        nfm=nfm, sys=sys, ctl_generic=ctl_generic, data=test_data,
        num_samples=num_samples_nf_eval, loss_fn=bounded_loss_fn, count_collisions=col_av
    )
    msg = 'Average loss: %.4f' % test_loss
    if col_av:
        msg += ' -- total number of collisions = %i' % test_num_col
    logger.info(msg)

    # plot closed-loop trajectories using the trained controller
    logger.info('Plotting closed-loop trajectories using the trained controller...')
    with torch.no_grad():
        z, _ = nfm.sample(100)
        z_mean = torch.mean(z, axis=0)
        _, xs_z_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z
        )
        _, xs_z_mean_plot = eval_norm_flow(
            sys=sys, ctl_generic=ctl_generic, data=plot_data,
            loss_fn=bounded_loss_fn, count_collisions=False, return_traj=True, params=z_mean
        )
        plot_trajectories(
            torch.cat((xs_z_plot[:, :5, :, :].squeeze(0), xs_z_mean_plot), 0),
            sys.xbar, sys.n_agents, text='CL - trained flow', T=t_ext,
            save_folder=save_folder, filename='CL_trained.pdf',
            obstacle_centers=bounded_loss_fn.obstacle_centers,
            obstacle_covs=bounded_loss_fn.obstacle_covs,
            plot_collisions=True, min_dist=bounded_loss_fn.min_dist
        )

