import torch, os
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors

from loss_functions.robots_loss_multi_batch import count_collisions

def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100,
    axis=False, f=5,
    obstacle_centers=None, obstacle_covs=None,
    # mark collisions between trajectories
    plot_collisions=False, min_dist=None,
):
    x = x.detach().cpu()
    # reshape to (batch_size, T, dim_states)
    if x.ndim==2:
        x = x.reshape(1, *x.shape)
    num_trajs = x.shape[0]

    # flatten xbar
    xbar=xbar.squeeze()

    filename = 'trajectories.pdf' if filename == '' else filename

    fig, ax = plt.subplots(figsize=(f,f))
    ax.set_title(text)
    palets = [sns.color_palette('dark:b_r', as_cmap=True), sns.color_palette('dark:salmon_r', as_cmap=True)]
    norm = mpl.colors.Normalize(vmin=3, vmax=3+num_trajs) # avoid too light and too dark colors
    cmaps = [mpl.cm.ScalarMappable(norm=norm, cmap=p) for p in palets]

    # plot trajectories
    for i in range(n_agents):
        plot_final = True
        # with sns.color_palette(palets[i%2], n_colors=num_trajs):
        for traj_num in range(num_trajs):
            color=cmaps[i].to_rgba(traj_num+1)
            # trajectory until T
            ax.plot(
                x[traj_num, :T+1, 4*i], x[traj_num, :T+1, 4*i+1],
                linewidth=1 if traj_num==num_trajs-1 else 0.1, color=color
            )
            # trajectory beyond T
            ax.plot(
                x[traj_num, T:, 4*i], x[traj_num, T:, 4*i+1],
                color='k', linewidth=0.3, linestyle='dotted', dashes=(3, 15)
            )
            # mark initial state
            ax.plot(
                x[traj_num, 0, 4*i], x[traj_num, 0, 4*i+1],
                marker='8', color=color
            )
            # mark final state
            if plot_final:
                ax.plot(
                    xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
                    marker='*', markersize=10, color=color
                )
                plot_final = False

    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(*ax.get_xlim(), 100), np.linspace(*ax.get_xlim(), 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

    # plot collisions
    if plot_collisions:
        assert not min_dist is None
        for traj_num in range(num_trajs):
            num_cols, col_matrix = count_collisions(
                x[traj_num, :, :].reshape(1, *x.shape[1:], 1),
                n_agents=n_agents, min_dist=min_dist, return_col_mat=True
            )
            if num_cols>0:
                col_matrix = col_matrix.squeeze(0)  # shape (T, n_agents, n_agents)
                for t in range(col_matrix.shape[0]):
                    for i in range(n_agents):
                        for j in range(i, n_agents):
                            if col_matrix[t, i, j]:
                                #define random color
                                col = (np.random.random(), np.random.random(), np.random.random())
                                ax.scatter(
                                    x[traj_num, t, 4*i], x[traj_num, t, 4*i+1],
                                    color=col, marker='x'
                                )
                                ax.scatter(
                                    x[traj_num, t, 4*j], x[traj_num, t, 4*j+1],
                                    color=col, marker='x'
                                )

                        # if col_matrix[t, i, i:].sum().item() > 0:
                        #     ax.scatter(
                        #         x[traj_num, t, 4*i], x[traj_num, t, 4*i+1],
                        #         color='k', marker='x'
                        #     )

    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format='pdf'
        )
    else:
        plt.show()
    plt.close()


def plot_traj_vs_time(t_end, n_agents, save_folder, x, u=None, text="", save=True, filename=''):
    filename = filename if filename=='' else filename+'_'
    now = datetime.now()
    formatted_date = now.strftime('%m-%d-%H:%M')
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i].detach().cpu())
        plt.plot(t, x[:,4*i+1].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2].detach().cpu())
        plt.plot(t, x[:,4*i+3].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i].detach().cpu())
            plt.plot(t, u[:, 2*i+1].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(
            os.path.join(
                save_folder,
                filename+text+'_x_u.pdf'
            ),
            format='pdf'
        )
    else:
        plt.show()
