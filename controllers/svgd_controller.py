# TODO: update
from numpy import random
import torch, time
from inference_algs.svgd import SVGD, RBF_Kernel, IMQSteinKernel


class SVGDCont():
    def __init__(
        self, gibbs_posteior, num_particles, logger,
        optimizer, lr, lr_decay=None,
        initial_particles=None, kernel='RBF', bandwidth=None
    ):
        """
        initial_particles: if None, init from the prior
        """
        # --- init ---
        self.logger = logger
        self.posterior = gibbs_posteior
        self.num_particles = num_particles

        self.best_particles = None
        self.fitted, self.over_fitted = False, False
        self.unknown_err = False

        """initialize particles"""
        if not initial_particles is None:
            # set given initial particles
            assert initial_particles.shape[0]==self.num_particles
            initial_sampled_particles = initial_particles.float()
            self.logger.info('[INFO] initialized particles with given value.')
        else:
            # sample initial particle locations from prior
            initial_sampled_particles = self.posterior.sample_params_from_prior(
                shape=(self.num_particles,)
            )
            self.logger.info('[INFO] initialized particles by sampling from the prior.')
        self.initial_particles = initial_sampled_particles
        # set particles to the initial value
        self.particles = self.initial_particles.detach().clone()
        self.particles.requires_grad = True

        """ Setup optimizer"""
        self._setup_optimizer(optimizer, lr, lr_decay)

        """ Setup SVGD inference"""
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

        # if lr_decay < 1.0:
        #     self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        # else:
        #     self.lr_scheduler = DummyLRScheduler()

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
    def fit(self, dataloader, epochs, over_fit_margin=None, cont_fit_margin=None, max_iter_fit=None,
            early_stopping=True, valid_data=None, log_period=500):
        """
        fits the hyper-posterior particles with SVGD

        Args:
            dataloader: (torch.utils.data.DataLoader) data loader for train disturbances - shape of each batch: (batch_size, T, num_states)
            over_fit_margin: abrupt training if slope of valid RMSE over one log_period > over_fit_margin (set to None
             to disable early stopping)
            cont_fit_margin: continue training for more iters if slope of valid RMSE over one log_period<-cont_fit_margin
            max_iter_fit: max iters to extend training
            early_stopping: return model at an evaluated iteration with the lowest valid RMSE
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

        if early_stopping:
            self.best_particles = None
            min_criterion = 1e6


        # initial evaluation on train data
        message = 'Iter %d/%d' % (0, epochs)
        if valid_data is not None:
            # initial evaluation on validation data
            valid_results = [self.eval_rollouts(valid_data)]
            message += ', Valid Loss: {:2.4f}'.format(valid_results[0])
        self.logger.info(message)

        last_params = self.particles.detach().clone()  # params in the last iteration

        t = time.time()
        itr = 1
        while itr <= epochs:
            sel_inds = random.RandomState(5).choice(valid_data.shape[0], size=5)
            task_dict_batch = valid_data[sel_inds, :, :]
            # --- take a step ---
            self.svgd.step(self.particles, task_dict_batch)
            last_params = self.particles.detach().clone()
            # try:
            #     print(self.particles.shape, task_dict_batch.shape)
            #     self.svgd.step(self.particles, task_dict_batch)
            #     last_params = self.particles.detach().clone()
            # except Exception as e:
            #     self.logger.info('[Unhandled ERR] in SVGD step: ' + type(e).__name__ + '\n')
            #     self.logger.info(e)
            #     self.unknown_err = True
            # epoch_loss = 0
            # for data_batch in dataloader:
            #     # --- take a step ---
            #     try:
            #         self.svgd.step(self.particles, data_batch)
            #         epoch_loss += self.svgd.log_prob_particles.item()
            #     except Exception as e:
            #         self.logger.info('[Unhandled ERR] in SVGD step: ' + type(e).__name__ + '\n')
            #         self.logger.info(e)
            #         self.unknown_err = True
                # --- update last params ---
                # last_params = self.particles.detach().clone()

            # --- print stats ---
            if (itr % log_period == 0) and (not self.unknown_err):
                print(self.particles.detach().clone()[0,0:10])
                duration = time.time() - t
                t = time.time()
                message = 'Epoch %d/%d - Time %.2f sec - SVGD Loss in epoch %.4f' % (
                    itr, epochs, duration, self.svgd.log_prob_particles
                )

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_data is not None:
                    # evaluate on validation set
                    try:
                        valid_res = self.eval_rollouts(valid_data)
                        valid_results.append(valid_res)
                        message +=  ', Valid Loss: {:2.4f}'.format(valid_res)
                    except Exception as e:
                        message += '[Unhandled ERR] in eval valid rollouts:'
                        self.logger.info(e)
                        self.unknown_err = True

                    # check over-fitting
                    if not over_fit_margin is None:
                        if valid_results[-1]-valid_results[-2] >= over_fit_margin * log_period:
                            self.over_fitted = True
                            message += '\n[WARNING] model over-fitted'

                    # check continue training
                    if (not ((cont_fit_margin is None) or (max_iter_fit is None))) and (itr+log_period <= max_iter_fit) and (itr+log_period > epochs):
                        if valid_results[-1]-valid_results[-2] <= - abs(cont_fit_margin) * log_period:
                            epochs += log_period
                            message += '\n[Info] extended training'

                    # update the best particles if early_stopping
                    if early_stopping and itr > 1:
                        if valid_results[-1] < min_criterion:
                            # self.logger.info('update best particle according to '+'train' if valid_data is None else 'valid')
                            min_criterion = valid_results[-1]
                            self.best_particles = self.particles.detach().clone()

                # log learning rate
                # message += ', LR: '+str(self.lr_scheduler.get_last_lr())

                # log info
                self.logger.info(message)


            # go one iter back if non-psd
            if self.unknown_err:
                self.particles = last_params.detach().clone()  # set back params to the previous iteration

            # stop training
            if self.over_fitted or self.unknown_err:
                break

            # # update learning rate
            # self.lr_scheduler.step()
            # go to next iter
            itr = itr+1

        self.fitted = True if not self.unknown_err else False

        # set back to the best particles if early stopping
        if early_stopping and (not self.best_particles is None):
            self.particles = self.best_particles
