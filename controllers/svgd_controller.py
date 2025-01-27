import torch, time, os
import matplotlib.pyplot as plt

from inference_algs.svgd import SVGD, RBF_Kernel, IMQSteinKernel


class SVGDCont():
    def __init__(
        self, gibbs_posterior, num_particles, logger,
        optimizer, lr, lr_decay=None,
        initial_particles=None, kernel='RBF', bandwidth=None
    ):
        """
        Initializes the SVGDController.

        Args:
            gibbs_posterior: The Gibbs posterior object.
            num_particles: Number of particles.
            logger: Logger object for logging information.
            optimizer: Optimizer to use.
            lr: Learning rate.
            lr_decay: Learning rate decay.
            initial_particles: Initial particles, if None, drawn from the prior.
            kernel: SVGD kernel type, default is 'RBF'.
            bandwidth: Bandwidth for the SVGD kernel. None or -1 for heuristic method.
        """
        # --- init ---
        self.logger = logger
        self.posterior = gibbs_posterior
        self.num_particles = num_particles

        self.best_particles = None
        self.fitted = False
        self.unknown_err = False

        # Initialize particles
        if initial_particles is not None:
            assert initial_particles.shape[0] == self.num_particles
            initial_sampled_particles = initial_particles.float()
            self.logger.info('[INFO] initialized particles with given value.')
        else:
            initial_sampled_particles = self.posterior.sample_params_from_prior(shape=(self.num_particles,))
            self.logger.info('[INFO] initialized particles by sampling from the prior.')

        self.initial_particles = initial_sampled_particles
        self.particles = self.initial_particles.detach().clone()
        self.particles.requires_grad = True

        # Setup optimizer
        self._setup_optimizer(optimizer, lr, lr_decay)

        # Setup SVGD inference
        if kernel == 'RBF':
            kernel = RBF_Kernel(bandwidth=bandwidth)
        elif kernel == 'IMQ':
            kernel = IMQSteinKernel(bandwidth=bandwidth)
        else:
            raise NotImplementedError

        self.svgd = SVGD(self.posterior, kernel, optimizer=self.optimizer)
    
    def _setup_optimizer(self, optimizer, lr, lr_decay):
        assert hasattr(self, 'particles'), "SVGD must be initialized before setting up optimizer"
        assert self.particles.requires_grad

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam([self.particles], lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD([self.particles], lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')


    def rollout(self, data):
        """
        rollout using current particles.
        Tracks grads.
        """
        # if len(data.shape)==2:
        #     data = torch.reshape(data, (data, *data.shape))

        res_xs, res_ys, res_us = [], [], []
        for particle_num in range(self.num_particles):
            particle = self.particles[particle_num, :]
            # set this particle as params of a controller
            cl_system = self.posterior.get_forward_cl_system(particle)
            # rollout
            xs, ys, us = cl_system.rollout(data)
            res_xs.append(xs)
            res_ys.append(ys)
            res_us.append(us)
        assert len(res_xs) == self.num_particles
        return res_xs, res_ys, res_us

    def eval_rollouts(self, data, get_full_list=False, loss_fn=None):
        """
        evaluates several rollouts given by 'data'.
        if 'get_full_list' is True, returns a list of losses for each particle.
        o.w., returns average loss of all particles.
        if 'loss_fn' is None, uses the bounded loss function as in Gibbs posterior.
        loss_fn can be provided to evaluate the dataset using the original unbounded loss.
        """
        with torch.no_grad():
            losses=[None]*self.num_particles
            res_xs, _, res_us = self.rollout(data)
            for particle_num in range(self.num_particles):
                if loss_fn is None:
                    losses[particle_num] = self.posterior.loss_fn.forward(
                        res_xs[particle_num], res_us[particle_num]
                    ).item()
                else:
                    losses[particle_num] = loss_fn.forward(
                        res_xs[particle_num], res_us[particle_num]
                    ).item()
        if get_full_list:
            return losses
        else:
            return sum(losses)/self.num_particles

    # ---------- FIT ----------
    def fit(self, train_dataloader, epochs, 
            early_stopping, tol_percentage=None, n_logs_no_change=None,
            return_best=True, valid_data=None, log_period=500, loss_fn=None, save_folder=None):
        """
        fits the hyper-posterior particles with SVGD

        Args:
            train_dataloader: (torch.utils.data.DataLoader) data loader for train disturbances - shape of each batch: (batch_size, T, num_states)
            return_best: return model at an evaluated iteration with the lowest valid RMSE
            valid_data: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            log_period (int) number of steps after which to print stats
        """
        # check data shape
        # if batch_size < 1:
        #     self.batch_size = self.train_d.shape[0]
        # else:
        #     self.batch_size = min(batch_size, self.train_d.shape[0])

        # (S, T, num_states) = data.shape
        # assert T == loss.T

        if return_best:
            self.best_particles = None
            min_valid_res = 1e6
        if early_stopping:
            assert not tol_percentage is None, 'Tolerance percentage must be provided for early stopping'
            assert not n_logs_no_change is None, 'Number of logs with no change must be provided for early stopping'
            assert tol_percentage >= 0 and tol_percentage < 1, 'Tolerance percentage must be in [0, 1)'
        if early_stopping or return_best:
            assert not valid_data is None, 'Validation data must be provided for early stopping or return_best'

        # queue of validation losses for early stopping
        if early_stopping:
            valid_imp_queue = [100]*n_logs_no_change   # don't stop at the beginning

        last_params = self.particles.detach().clone()  # params in the last iteration

        t = time.time()
        svgd_loss_hist = [None]*epochs
        for epoch in range(1+epochs):
            # iterate over all data batches
            for train_data_batch in train_dataloader:
                # take a step
                try:
                    self.svgd.step(self.particles, train_data_batch)
                except Exception as e:
                    self.logger.info('[Unhandled ERR] in SVGD step: ' + type(e).__name__ + '\n')
                    self.logger.info(e)
                    self.unknown_err = True
            
            last_params = self.particles.detach().clone()
            svgd_loss_hist[epoch]= self.svgd.log_prob_particles.item() #to('cpu').data.numpy()

            # --- print stats ---
            if (epoch % log_period == 0) and (not self.unknown_err):
                # print(self.particles.detach().clone()[0,0:10])
                duration = time.time() - t
                t = time.time()
                message = 'Epoch %d/%d - Time %.2f sec - SVGD Loss in epoch %.4f' % (
                    epoch, epochs, duration, self.svgd.log_prob_particles
                )

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_data is not None:
                    # evaluate on validation set
                    try:
                        valid_res = self.eval_rollouts(valid_data, loss_fn=loss_fn)
                        message +=  ', Valid Loss: {:2.4f}'.format(valid_res)
                    except Exception as e:
                        message += '[Unhandled ERR] in eval valid rollouts:'
                        self.logger.info(e)
                        self.unknown_err = True

                    # update the best particles if return_best
                    imp = 100 * (min_valid_res-valid_res)/min_valid_res
                    if return_best and epoch > 1:
                        if valid_res < min_valid_res:
                            min_valid_res = valid_res
                            self.best_particles = self.particles.detach().clone()

                    # early stopping
                    if early_stopping:
                        # add the current valid loss to the queue
                        valid_imp_queue.pop(0)
                        valid_imp_queue.append(imp)
                        # check if there is no improvement
                        if all([valid_imp_queue[i] <tol_percentage for i in range(n_logs_no_change)]):
                            message += ' ---||--- early stopping at epoch %i' % (epoch)
                            self.logger.info(message)
                            break

                # log info
                self.logger.info(message)

                # plot loss
                if not save_folder is None:
                    plt.figure(figsize=(10, 10))
                    plt.plot(svgd_loss_hist[:epoch+1], label='loss')
                    plt.legend()
                    plt.savefig(os.path.join(save_folder, 'loss.pdf'))
                    plt.show()


            # go one iter back if non-psd
            if self.unknown_err:
                self.logger.info('Going back one iteration due to unknown error.')
                self.particles = last_params.detach().clone()  # set back params to the previous iteration
                break

        self.fitted = True

        # set back to the best particles if early stopping
        if return_best and (not self.best_particles is None):
            self.particles = self.best_particles
