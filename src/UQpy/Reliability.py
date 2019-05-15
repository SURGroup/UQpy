# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for all the reliability methods supported in UQpy."""

from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC
from UQpy.Transformations import *

########################################################################################################################
########################################################################################################################
#                                        Subset Simulation
########################################################################################################################


class SubsetSimulation:
    """
        Description:

            A class used to perform Subset Simulation.

            This class estimates probability of failure for a user-defined model using Subset Simulation

            References:
            S.-K. Au and J. L. Beck, “Estimation of small failure probabilities in high dimensions by
            subset simulation,” Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.

        Input:
            :param dimension:  A scalar value defining the dimension of target density function.
                            Default: 1
            :type dimension: int

            :param nsamples_ss: Number of samples to generate in each conditional subset
                                No Default Value: nsamples_ss must be prescribed
            :type nsamples_ss: int

            :param p_cond: Conditional probability at each level
                                Default: p_cond = 0.1
            :type p_cond: float

            :param pdf_proposal_type, pdf_proposal_scale, pdf_target, log_pdf_target, pdf_target_params, jump, nburn,
                    pdf_target_copula, pdf_target_copula_params, pdf_target_type, algorithm: See MCMC in SampleMethods

            :param model_script, model_object_name, input_template, var_names, output_script, output_object_name,
                       ntasks, cores_per_task, nodes, resume, verbose, model_dir, cluster: See RunModel class.

    Output:

    :return self.pf: Probability of failure estimate
    :rtype self.pf: float
    :return self.cov1: Coefficient of variation - Au & Beck, Independent Chains
    :rtype self.cov1: float
    :return self.cov2: Coefficient of variation - New Dependent Chains
    :rtype self.cov2: float
    """

    # Authors: Dimitris G.Giovanis, Michael D. Shields
    # Last Modified: 4/7/19 by Dimitris G. Giovanis

    def __init__(self, dimension=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 pdf_target=None, log_pdf_target=None, pdf_target_params=None, pdf_target_copula=None,
                 pdf_target_copula_params=None, pdf_target_type='joint_pdf', seed=None,
                 algorithm='MH', jump=1, nsamples_ss=None, nburn=0, samples_init=None, p_cond=None,
                 verbose=False,  model_script=None, model_object_name=None, input_template=None, var_names=None,
                 output_script=None, output_object_name=None, n_tasks=1, cores_per_task=1, nodes=1, resume=False,
                 model_dir=None, cluster=False):

        self.dimension = dimension
        self.log_pdf_target = log_pdf_target
        self.pdf_target_copula = pdf_target_copula
        self.samples_init = samples_init
        self.pdf_target_copula_params = pdf_target_copula_params
        self.jump = jump
        self.nburn = nburn
        self.verbose = verbose
        self.pdf_target_type = pdf_target_type
        self.pdf_target = pdf_target
        self.pdf_target_params = pdf_target_params
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.nsamples_ss = nsamples_ss
        self.algorithm = algorithm

        self.input_template = input_template
        self.var_names = var_names
        self.model_script = model_script
        self.model_object_name = model_object_name
        self.output_object_name = output_object_name
        self.n_tasks = n_tasks
        self.cluster = cluster
        self.model_dir = model_dir
        self.output_script = output_script
        self.cores_per_task = cores_per_task
        self.nodes = nodes
        self.resume = resume
        self.p_cond = p_cond
        self.g = list()
        self.samples = list()
        self.g_level = list()
        self.d12 = list()
        self.d22 = list()
        if seed is None:
            self.seed = np.zeros(self.dimension)
        else:
            self.seed = seed
        # Hard-wire the maximum number of conditional levels.
        self.max_level = 20

        # Initialize variables and perform error checks
        self.init_sus()

        # Select the appropriate Subset Simulation Algorithm
        if self.algorithm == 'MMH':
            print('UQpy: Running Subset Simulation....')
            [self.pf, self.cov1, self.cov2] = self.run_subsim_mmh()
        elif self.algorithm == 'Stretch':
            [self.pf, self.cov1, self.cov2] = self.run_subsim_stretch()
        print('Done!')

    # Run Subset Simulation using Modified Metropolis Hastings
    def run_subsim_mmh(self):
        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)

        # Generate the initial samples - Level 0
        if self.samples_init is None:
            x_init = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                          pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
                          log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
                          pdf_target_copula=self.pdf_target_copula,
                          pdf_target_copula_params=self.pdf_target_copula_params,
                          pdf_target_type=self.pdf_target_type,
                          algorithm=self.algorithm, jump=self.jump, nsamples=self.nsamples_ss, seed=self.seed,
                          nburn=self.nburn, verbose=self.verbose)

            self.samples.append(x_init.samples)
        else:
            self.samples.append(self.samples_init)

        g_init = RunModel(samples=self.samples[step], model_script=self.model_script,
                          model_object_name=self.model_object_name,
                          input_template=self.input_template, var_names=self.var_names,
                          output_script=self.output_script,
                          output_object_name=self.output_object_name,
                          ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes, resume=self.resume,
                          verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

        self.g.append(np.asarray(g_init.qoi_list).reshape(-1,))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self.cov_sus(step)
        self.d12.append(d1 ** 2)
        self.d22.append(d2 ** 2)

        while self.g_level[step] > 0 and step < self.max_level:

            step = step + 1
            self.samples.append(self.samples[step - 1][g_ind[0:n_keep], :])
            self.g.append(self.g[step - 1][g_ind[:n_keep]])

            for i in range(self.nsamples_ss-n_keep):
                x0 = self.samples[step][i]

                x_mcmc = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                              pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
                              log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
                              pdf_target_copula=self.pdf_target_copula,
                              pdf_target_copula_params=self.pdf_target_copula_params,
                              pdf_target_type=self.pdf_target_type,
                              algorithm= self.algorithm, jump=self.jump, nsamples=2, seed=x0,
                              nburn=self.nburn, verbose=self.verbose)

                x_temp = x_mcmc.samples[1].reshape((1, self.dimension))

                g_model = RunModel(samples=x_temp, model_script=self.model_script,
                                   model_object_name=self.model_object_name,
                                   input_template=self.input_template, var_names=self.var_names,
                                   output_script=self.output_script,
                                   output_object_name=self.output_object_name,
                                   ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
                                   resume=self.resume,
                                   verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

                g_temp = g_model.qoi_list

                # Accept or reject the sample
                if g_temp < self.g_level[step - 1]:
                    self.samples[step] = np.vstack((self.samples[step], x_temp))
                    self.g[step] = np.hstack((self.g[step], g_temp[0]))
                else:
                    self.samples[step] = np.vstack((self.samples[step], self.samples[step][i]))
                    self.g[step] = np.hstack((self.g[step], self.g[step][i]))

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])
            # Estimate coefficient of variation of conditional probability of first level
            d1, d2 = self.cov_sus(step)
            self.d12.append(d1 ** 2)
            self.d22.append(d2 ** 2)

        n_fail = len([value for value in self.g[step] if value < 0])

        pf = self.p_cond**step*n_fail/self.nsamples_ss
        cov1 = np.sqrt(np.sum(self.d12))
        cov2 = np.sqrt(np.sum(self.d22))

        return pf, cov1, cov2

    # Run Subset Simulation using the Affine Invariant Ensemble Sampler
    def run_subsim_stretch(self):
        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)

        # Generate the initial samples - Level 0
        if self.samples_init is None:
            x_init = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                          pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
                          log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
                          pdf_target_copula=self.pdf_target_copula,
                          pdf_target_copula_params=self.pdf_target_copula_params,
                          pdf_target_type=self.pdf_target_type,
                          algorithm='MMH', jump=self.jump, nsamples=self.nsamples_ss, seed=self.seed,
                          nburn=self.nburn, verbose=self.verbose)
            self.samples.append(x_init.samples)
        else:
            self.samples.append(self.samples_init)

        g_init = RunModel(samples=self.samples[step], model_script=self.model_script,
                          model_object_name=self.model_object_name,
                          input_template=self.input_template, var_names=self.var_names,
                          output_script=self.output_script,
                          output_object_name=self.output_object_name,
                          ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes, resume=self.resume,
                          verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

        self.g.append(np.asarray(g_init.qoi_list))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self.cov_sus(step)
        self.d12.append(d1 ** 2)
        self.d22.append(d2 ** 2)

        while self.g_level[step] > 0:

            step = step + 1
            self.samples.append(self.samples[step - 1][g_ind[0:n_keep]])
            self.g.append(self.g[step - 1][g_ind[:n_keep]])

            for i in range(self.nsamples_ss - n_keep):

                x0 = self.samples[step][i:i+n_keep]

                x_mcmc = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                              pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
                              log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
                              pdf_target_copula=self.pdf_target_copula,
                              pdf_target_copula_params=self.pdf_target_copula_params,
                              pdf_target_type=self.pdf_target_type,
                              algorithm= self.algorithm, jump=self.jump, nsamples=n_keep+1, seed=x0,
                              nburn=self.nburn, verbose=self.verbose)

                x_temp = x_mcmc.samples[n_keep].reshape((1, self.dimension))
                g_model = RunModel(samples=x_temp, model_script=self.model_script,
                                   model_object_name=self.model_object_name,
                                   input_template=self.input_template, var_names=self.var_names,
                                   output_script=self.output_script,
                                   output_object_name=self.output_object_name,
                                   ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
                                   resume=self.resume,
                                   verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

                g_temp = g_model.qoi_list

                # Accept or reject the sample
                if g_temp < self.g_level[step - 1]:
                    self.samples[step] = np.vstack((self.samples[step], x_temp))
                    self.g[step] = np.hstack((self.g[step], g_temp[0]))
                else:
                    self.samples[step] = np.vstack((self.samples[step], self.samples[step][i]))
                    self.g[step] = np.hstack((self.g[step], self.g[step][i]))

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])
            d1, d2 = self.cov_sus(step)
            self.d12.append(d1 ** 2)
            self.d22.append(d2 ** 2)

        n_fail = len([value for value in self.g[step] if value < 0])
        pf = self.p_cond ** step * n_fail / self.nsamples_ss
        cov1 = np.sqrt(np.sum(self.d12))
        cov2 = np.sqrt(np.sum(self.d22))

        return pf, cov1, cov2

    def init_sus(self):

        # Set default dimension to 1
        if self.dimension is None:
            self.dimension = 1

        # Check that the number of samples per subset is defined.
        if self.nsamples_ss is None:
            raise NotImplementedError('Number of samples per subset not defined. This is required.')

        # Check that the MCMC algorithm is properly defined.
        if self.algorithm is None:
            self.algorithm = 'MMH'
        elif self.algorithm not in ['Stretch', 'MMH']:
            raise NotImplementedError('Invalid MCMC algorithm. Select from: MMH, Stretch')

        # Check that a valid conditional probability is specified.
        if type(self.p_cond).__name__ != 'float':
            raise NotImplementedError('Invalid conditional probability. p_cond must be of float type.')
        elif self.p_cond <= 0. or self.p_cond >= 1.:
            raise NotImplementedError('Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the user has defined a model
        if self.model_script is None:
            raise NotImplementedError('Subset Simulation requires the specification of a computational model. Please '
                                      'specify the model using the model_script input.')

    def cov_sus(self, step):
        n = self.g[step].size
        if step == 0:
            d1 = np.sqrt((1 - self.p_cond) / (self.p_cond * n))
            d2 = np.sqrt((1 - self.p_cond) / (self.p_cond * n))

            return d1, d2
        else:
            n_c = int(self.p_cond * n)
            n_s = int(1 / self.p_cond)
            indicator = np.reshape(self.g[step] < self.g_level[step], (n_s, n_c))
            gamma = self.corr_factor_gamma(indicator, n_s, n_c)
            # beta_hat, r_jn0 = self.corr_factor_beta(indicator, n_s, n_c, self.p_cond)  # Eq. 24
            g_temp = np.reshape(self.g[step], (n_s, n_c))
            # beta_hat, r_jn0 = self.corr_factor_beta(g_temp, n_s, n_c, self.p_cond)
            beta_hat, r_jn0 = self.corr_factor_beta(indicator, g_temp, n_s, n_c, self.p_cond)
            # beta_i = (n_c - 1) * r_jn0 + beta_hat
            beta_i = r_jn0 + beta_hat

            d1 = np.sqrt(((1 - self.p_cond) / (self.p_cond * n)) * (1 + gamma))
            d2 = np.sqrt(((1 - self.p_cond) / (self.p_cond * n)) * (1 + gamma + beta_i))

            return d1, d2

    def corr_factor_gamma(self, indicator, n_s, n_c):

        gamma = np.zeros(n_s - 1)
        r = np.zeros(n_s)
        # n = n_c * n_s

        ii = indicator * 1
        r_ = ii @ ii.T / n_c - self.p_cond ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_s - 1):
            gamma[i] = (1 - ((i + 1) / n_s)) * r[i+1]
        gamma = 2 * np.sum(gamma)

        # gamma = np.zeros(n_s - 1)
        # r = np.zeros(n_s - 1)
        # n = n_c * n_s
        #
        # sums = 0
        # for k in range(n_c):
        #     for ip in range(n_s):
        #         sums = sums + (indicator[ip, k] * indicator[ip, k])  # sums inside (Ref. 1 Eq. 22)
        #
        # r_0 = (1 / n) * sums - self.p_cond ** 2
        #
        # for i in range(n_s - 1):
        #     z = 0
        #     for k in range(n_c):
        #         for ip in range(n_s - i - 1):
        #             z = z + (indicator[ip, k] * indicator[ip + i + 1, k])
        #
        #     r[i] = (1 / (n - (i + 1) * n_c)) * z - self.p_cond ** 2
        #     gamma[i] = (1 - ((i + 1) / n_s)) * (r[i] / r_0)
        #
        # gamma = 2 * np.sum(gamma)

        return gamma

    # def corr_factor_beta(self, indicator, n_s, n_c, p_cond):
    #
    #     beta = np.zeros(n_s - 1)
    #     r_jn = np.zeros(n_s)
    #     I = indicator * 1
    #
    #     for i in range(n_s):
    #         Rx = I[0:n_s-i, :].T @ I[i:, :]
    #         # np.fill_diagonal(Rx, 0)
    #         r_jn[i] = np.sum(Rx) / ((n_c * (n_c)) * (n_s - i)) - p_cond ** 2
    #         # r_jn[i] = np.sum(Rx) / ((n_c * n_c) * (n_s - i)) - p_cond ** 2
    #     r0 = p_cond * (1 - p_cond)
    #     r_jn = r_jn / r0
    #
    #     for k in range(n_s - 1):
    #         beta[k] = (1 - ((k + 1) / n_s)) * (r_jn[k]) * r_jn[0]
    #
    #     beta = 2 * (n_c - 1) * np.sum(beta)
    #
    #     return beta, r_jn[0]

    # Operate only on pairs with same starting point
    def corr_factor_beta(self, indicator, g, n_s, n_c, p_cond):

        r_jn = 0
        u = np.unique(g[0, :])
        for i in range(u.size):
            ii = indicator[:, g[0, :] == u[i]]
            r_jn = r_jn + ii.shape[1]*(ii.shape[1]-1)/2

        beta = 0

        # total = 0
        # r_jn = 0
        # if U.size < n_c:
        #     for i in range(U.size):
        #         I = indicator[:, g[0, :] == U[i]]
        #         I = I * 1
        #
        #         r_temp = I.T @ I
        #         r0 = np.sum(r_temp) / (((r_temp.shape[0] * r_temp.shape[0])) * n_s) - p_cond ** 2
        #         r0 = r0 * r_temp.shape[0] * (r_temp.shape[0] - 1) / 2
        #         r0 = r0
        #         r_jn = r_jn + r0
        #         # r = r_temp / (I.shape[1] * (I.shape[1]-1))
        #         total = total + I.shape[1]
        #
        #     r_jn = r_jn / total
        #     print(r_jn)
        #
        # # for i in range(n_c):
        #
        #
        # beta = np.zeros(n_s - 1)
        # r_jn = np.zeros(n_s)
        # I = g
        #
        # for i in range(n_s):
        #     Rx = I[0:n_s-i, :].T @ I[i:, :]
        #     np.fill_diagonal(Rx, 0)
        #     r_jn[i] = np.sum(Rx) / ((n_c * (n_c-1)) * (n_s - i)) - np.mean(g) ** 2
        #     print(r_jn)
        #     # r_jn[i] = np.sum(Rx) / ((n_c * n_c) * (n_s - i)) - p_cond ** 2
        # r0 = np.var(g)
        # r_jn = r_jn / r0
        #
        # for k in range(n_s - 1):
        #     beta[k] = (1 - ((k + 1) / n_s)) * (r_jn[k]) * r_jn[0]
        #
        # beta = 2 * (n_c - 1) * np.sum(beta)

        return beta, r_jn



    # def corr_factor_beta(self, g, n_s, n_c, p_cond):
    #
    #     beta = np.zeros(n_s - 1)
    #     r_jn = np.zeros(n_s)
    #     I = g
    #
    #     for i in range(n_s):
    #         Rx = I[0:n_s-i, :].T @ I[i:, :]
    #         np.fill_diagonal(Rx, 0)
    #         r_jn[i] = np.sum(Rx) / ((n_c * (n_c-1)) * (n_s - i)) - np.mean(g) ** 2
    #         print(r_jn)
    #         # r_jn[i] = np.sum(Rx) / ((n_c * n_c) * (n_s - i)) - p_cond ** 2
    #     r0 = np.var(g)
    #     r_jn = r_jn / r0
    #
    #     for k in range(n_s - 1):
    #         beta[k] = (1 - ((k + 1) / n_s)) * (r_jn[k]) * r_jn[0]
    #
    #     beta = 2 * (n_c - 1) * np.sum(beta)
    #
    #     return beta, r_jn[0]


    # Version where cross-correlations are computed from g
    # def corr_factor_beta(self, g, n_s, n_c, p_cond):
    #
    #     beta = np.zeros(n_s - 1)
    #     r_jn = np.zeros(n_s )
    #     # n = n_c * n_s
    #     # factor = scipy.misc.comb(n_c,2)
    #
    #     # sums = 0
    #     # for j in range(n_c):
    #     #     for n_ in range(n_c):
    #     #         for l in range(n_s):
    #     #             if n_ != j:
    #     #                 sums = sums + (indicator[l, n_] * indicator[l, j])
    #     # I = indicator*1
    #     # R1 =  np.dot(np.transpose(I), I)/10 - p_cond**2
    #
    #     mu_g = np.mean(g)
    #     R = np.dot(g, g.T)/n_c - mu_g**2
    #     for i in range(R.shape[0]):
    #         r_jn[i] = np.sum(np.diag(R,i))/(R.shape[0]-i)
    #     # R0 = p_cond*(1-p_cond)
    #     R0 = np.var(g)
    #     r_jn = r_jn/R0
    #     # s1 = np.sum(np.dot(np.transpose(I), I))
    #     # s2 = np.sum(np.dot(np.transpose(I), I)) - np.sum(np.diag(np.dot(np.transpose(I), I)))
    #     # np.mean(R1)
    #
    #     # r_jn0 = (1 / n) * sums - self.p_cond ** 2
    #     # r_jn0 = 1 / (factor - n_c) * (1 / (n / n_c)) * sums - self.p_cond ** 2
    #
    #     for k in range(n_s - 1):
    #         # z = 0
    #         # for j in range(n_c):
    #         #     for n_ in range(n_c - k):
    #         #         for l in range(n_s - k - 1):
    #         #             if n_ != j:
    #         #                 z = z + (indicator[l, j] * indicator[l + k + 1, n_])
    #         #
    #         # r_jn[k] = 1 / (factor - n_c) * (1 / (n - (k + 1) * n_c)) * z - self.p_cond ** 2
    #         beta[k] = (1 - ((k + 1) / n_s)) * (r_jn[k])*R0
    #
    #     beta = 2 * (n_c - 1) * np.sum(beta)
    #     # beta = 2 * np.sum(beta)
    #
    #     return beta, r_jn[0]

        # Version where cross-correlations are computed just from indicator
    # def corr_factor_beta(self, indicator, n_s, n_c, p_cond):
    #
    #         beta = np.zeros(n_s - 1)
    #         r_jn = np.zeros(n_s)
    #         # n = n_c * n_s
    #         # factor = scipy.misc.comb(n_c,2)
    #
    #         # sums = 0
    #         # for j in range(n_c):
    #         #     for n_ in range(n_c):
    #         #         for l in range(n_s):
    #         #             if n_ != j:
    #         #                 sums = sums + (indicator[l, n_] * indicator[l, j])
    #         I = indicator * 1
    #         # R1 =  np.dot(np.transpose(I), I)/10 - p_cond**2
    #
    #         R = np.dot(I, np.transpose(I)) / n_c - p_cond ** 2
    #         for i in range(R.shape[0]):
    #             r_jn[i] = np.sum(np.diag(R, i)) / (R.shape[0] - i)
    #         R0 = p_cond * (1 - p_cond)
    #         r_jn = r_jn / R0
    #         # s1 = np.sum(np.dot(np.transpose(I), I))
    #         # s2 = np.sum(np.dot(np.transpose(I), I)) - np.sum(np.diag(np.dot(np.transpose(I), I)))
    #         # np.mean(R1)
    #
    #         # r_jn0 = (1 / n) * sums - self.p_cond ** 2
    #         # r_jn0 = 1 / (factor - n_c) * (1 / (n / n_c)) * sums - self.p_cond ** 2
    #
    #         for k in range(n_s - 1):
    #             # z = 0
    #             # for j in range(n_c):
    #             #     for n_ in range(n_c - k):
    #             #         for l in range(n_s - k - 1):
    #             #             if n_ != j:
    #             #                 z = z + (indicator[l, j] * indicator[l + k + 1, n_])
    #             #
    #             # r_jn[k] = 1 / (factor - n_c) * (1 / (n - (k + 1) * n_c)) * z - self.p_cond ** 2
    #             beta[k] = (1 - ((k + 1) / n_s)) * (r_jn[k]) * R0
    #
    #         beta = 2 * (n_c - 1) * np.sum(beta)
    #         # beta = 2 * np.sum(beta)
    #
    #         # r_jn[0] = 0.
    #         return beta, r_jn[0]

        # def cov_sus(self, step):
        #     n = self.g[step].size
        #     if step == 0:
        #         di = np.sqrt((1 - self.p_cond) / (self.p_cond * n))
        #     else:
        #         nc = int(self.p_cond * n)
        #         r_zero = self.p_cond * (1 - self.p_cond)
        #         index = np.zeros(n)
        #         index[np.where(self.g[step] < self.g_level[step])] = 1
        #         indices = np.zeros(shape=(int(n / nc), nc)).astype(int)
        #         for i in range(int(n / nc)):
        #             for j in range(nc):
        #                 if i == 0:
        #                     indices[i, j] = j
        #                 else:
        #                     indices[i, j] = indices[i - 1, j] + nc
        #         gamma = 0
        #         rho = np.zeros(int(n / nc) - 1)
        #         for k in range(int(n / nc) - 1):
        #             z = 0
        #             for j in range(int(nc)):
        #                 for l in range(int(n / nc) - k):
        #                     z = z + index[indices[l, j]] * index[indices[l + k, j]]
        #
        #             rho[k] = (1 / (n - k * nc) * z - self.p_cond ** 2) / r_zero
        #             gamma = gamma + 2 * (1 - k * nc / n) * rho[k]
        #
        #         di = np.sqrt((1 - self.p_cond) / (self.p_cond * n) * (1 + gamma))
        #
        #     return di

########################################################################################################################
########################################################################################################################
#                                        First/Second order reliability method
########################################################################################################################


class TaylorSeries:

    # Authors: Dimitris G.Giovanis
    # Last Modified: 11/19/18 by Dimitris G. Giovanis

    def __init__(self, dimension=None, dist_name=None, dist_params=None, n_iter=1000, corr=None, method=None, seed=None,
                 algorithm=None, model_script=None, model_object_name=None, input_template=None, var_names=None,
                 output_script=None, output_object_name=None, n_tasks=1, cores_per_task=1, nodes=1, resume=False,
                 verbose=False, model_dir=None, cluster=False):
        """
            Description: A class that performs reliability analysis of a model using the First Order Reliability Method
                         (FORM) and Second Order Reliability Method (SORM) that belong to the family of Taylor series
                         expansion methods.

            Input:
                :param dimension: Number of random variables
                :type dimension: int

                :param dist_name: Probability distribution model for each random variable (see Distributions class).
                :type dist_name: list/string

                :param dist_params: Probability distribution model parameters for each random variable.
                                   (see Distributions class).
                :type dist_params: list

                :param n_iter: Maximum number of iterations for the Hasofer-Lind algorithm
                :type n_iter: int

                :param seed: Initial seed
                :type seed: np.array

                :param corr: Correlation structure of the random vector (See Transformation class).
                :type corr: ndarray

                :param method: Method used for the reliability problem -- available methods: 'FORM', 'SORM'
                :type method: str

                :param algorithm: Algorithm used to solve the optimization problem -- available algorithms: 'HL'.
                :type algorithm: str

                :param model_script, model_object_name, input_template, var_names, output_script, output_object_name,
                       ntasks, cores_per_task, nodes, resume, verbose, model_dir, cluster: See RunModel class.
        """
        self.dimension = dimension
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.n_iter = n_iter
        self.method = method
        self.algorithm = algorithm
        self.seed = seed
        self.model_object_name = model_object_name
        self.model_script = model_script
        self.output_object_name = output_object_name
        self.input_template = input_template
        if var_names is None:
            var_names = ['dummy'] * self.dimension
        if corr is None:
            corr = np.eye(dimension)
        self.corr = corr
        self.var_names = var_names
        self.n_tasks = n_tasks
        self.cores_per_task = cores_per_task
        self.nodes = nodes
        self.resume = resume
        self.verbose = verbose
        self.model_dir = model_dir
        self.cluster = cluster
        self.output_script = output_script

        if self.method == 'FORM':
            print('Running FORM...')
        elif self.method == 'SORM':
            print('Running SORM...')

        if self.algorithm == 'HL':
            [self.DesignPoint_U, self.DesignPoint_X, self.HL_beta, self.Prob_FORM,
             self.Prob_SORM, self.iterations] = self.form_hl()

        '''
        print('Design point in standard normal space: %s' % self.DesignPoint_Z)
        print('Design point in original space: %s' % self.DesignPoint_X)
        print('Hasofer-Lind reliability index: %s' % self.HL_ReliabilityIndex)
        print('FORM probability of failure: %s' % self.ProbabilityOfFailure_FORM)

        if self.method == 'SORM':
            print('SORM probability of failure: %s' % self.ProbabilityOfFailure_SORM)

        print('Total number of function calls: %s' % self.iterations)
        '''

    def form_hl(self):
        n = self.dimension  # number of random variables (dimension)
        # initialization
        max_iter = self.n_iter
        tol = 1e-5
        u = np.zeros([max_iter + 1, n])
        if self.seed is not None:
            u[0, :] = Nataf(dimension=self.dimension, input_samples=self.seed.reshape(1, -1),
                            dist_name=self.dist_name, dist_params=self.dist_params, corr=self.corr).samples
        x = np.zeros_like(u)
        beta = np.zeros(max_iter)
        converge_ = False

        for k in range(max_iter):
            # transform the initial point in the original space:  U to X
            u_x = InvNataf(dimension=self.dimension, input_samples=u[k, :].reshape(1, -1),
                           dist_name=self.dist_name, dist_params=self.dist_params, corr_norm=self.corr)

            x[k, :] = u_x.samples
            jacobian = u_x.jacobian[0]
            # 1. evaluate Limit State Function at point

            g = RunModel(samples=x[k, :].reshape(1, -1), model_script=self.model_script,
                         model_object_name=self.model_object_name,
                         input_template=self.input_template, var_names=self.var_names, output_script=self.output_script,
                         output_object_name=self.output_object_name,
                         ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes, resume=self.resume,
                         verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

            # 2. evaluate Limit State Function gradient at point u_k and direction cosines
            dg = gradient(sample=x[k, :].reshape(1, -1), dimension=self.dimension, eps=0.1,
                          model_script=self.model_script,
                          model_object_name=self.model_object_name,
                          input_template=self.input_template, var_names=self.var_names,
                          output_script=self.output_script,
                          output_object_name=self.output_object_name,
                          ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes, resume=self.resume,
                          verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster, order='second')
            try:
                p = np.linalg.solve(jacobian, dg[0, :])
            except:
                print('Bad transformation')
                if self.method == 'FORM':
                    u_star = np.inf
                    x_star = np.inf
                    beta = np.inf
                    pf = np.inf

                    return u_star, x_star, beta, pf, [], k

                elif self.method == 'SORM':
                    u_star = np.inf
                    x_star = np.inf
                    beta = np.inf
                    pf = np.inf
                    pf_srom = np.inf

                    return u_star, x_star, beta, pf, pf_srom, k

            try:
                np.isnan(p)
            except:

                print('Bad transformation')
                if self.method == 'FORM':
                    u_star = np.inf
                    x_star = np.inf
                    beta = np.inf
                    pf = np.inf

                    return u_star, x_star, beta, pf, [], k

                elif self.method == 'SORM':
                    u_star = np.inf
                    x_star = np.inf
                    beta = np.inf
                    pf = np.inf
                    pf_srom = np.inf

                    return u_star, x_star, beta, pf, pf_srom, k

            norm_grad = np.linalg.norm(p)
            alpha = p / norm_grad
            alpha = alpha.squeeze()
            # 3. calculate first order beta
            beta[k + 1] = -np.inner(u[k, :].T, alpha) + g.qoi_list[0] / norm_grad
            #-np.inner(u[k, :].T, alpha) + g.qoi_list[0] / norm_grad
            # 4. calculate u_{k+1}
            u[k + 1, :] = -beta[k + 1] * alpha
            # next iteration
            if np.linalg.norm(u[k + 1, :] - u[k, :]) <= tol:
                converge_ = True
                # delete unnecessary data
                u = u[:k + 1, :]
                # compute design point, reliability index and Pf
                u_star = u[-1, :]
                # transform points in the original space
                u_x = InvNataf(dimension=self.dimension, input_samples=u_star.reshape(1, -1),
                               dist_name=self.dist_name, dist_params=self.dist_params, corr_norm=self.corr)
                x_star = u_x.samples
                beta = beta[k]
                pf = stats.norm.cdf(-beta)
                if self.method == 'SORM':
                    k = 3 * (k+1) + 5
                    der_ = dg[1, :]
                    mixed_der = gradient(sample=x_star.reshape(1, -1), eps=0.1, dimension=self.dimension,
                                         model_script=self.model_script,
                                         model_object_name=self.model_object_name,
                                         input_template=self.input_template, var_names=self.var_names,
                                         output_script=self.output_script,
                                         output_object_name=self.output_object_name,
                                         ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
                                         resume=self.resume,
                                         verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster,
                                         order='mixed')

                    hessian = eval_hessian(self.dimension, mixed_der, der_)
                    q = np.eye(self.dimension)
                    q[:, 0] = u_star.T
                    q_, r_ = np.linalg.qr(q)
                    q0 = np.fliplr(q_)
                    a = np.dot(np.dot(q0.T, hessian), q0)
                    if self.dimension > 1:
                        jay = np.eye(self.dimension - 1) + beta * a[:self.dimension - 1,
                                                                    :self.dimension - 1] / norm_grad
                    elif self.dimension == 1:
                        jay = np.eye(self.dimension) + beta * a[:self.dimension, :self.dimension] / norm_grad
                    correction = 1 / np.sqrt(np.linalg.det(jay))
                    pf_srom = pf * correction

                    return u_star, x_star, beta, pf, pf_srom, k

                elif self.method == 'FORM':
                    k = 3 * (k + 1)
                    return u_star, x_star[0], beta, pf,  [], k
            else:
                continue

        if converge_ is False:
            print("{0} did not converge".format(self.method))

            if self.method == 'FORM':
                u_star = np.inf
                x_star = np.inf
                beta = np.inf
                pf = np.inf

                return u_star, x_star, beta, pf, [], k

            elif self.method == 'SORM':
                u_star = np.inf
                x_star = np.inf
                beta = np.inf
                pf = np.inf
                pf_srom = np.inf

                return u_star, x_star, beta, pf, pf_srom, k