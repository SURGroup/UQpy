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
import warnings

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

    def __init__(self, mcmc_object=None, runmodel_object=None, samples_init=None, p_cond=0.1, nsamples_ss=1000,
                 max_level=10, verbose=False):

        # Initialize internal attributes from information passed in
        self.mcmc_objects = [mcmc_object]
        self.runmodel_object = runmodel_object
        self.samples_init = samples_init
        self.p_cond = p_cond
        self.nsamples_ss = nsamples_ss
        self.max_level = max_level
        self.verbose = verbose

        # Perform initial error checks
        self.init_sus()

        # Initialize new attributes/variables
        self.samples = list()
        self.g = list()
        self.g_level = list()
        self.d12 = list()
        self.d22 = list()

        if self.verbose:
            if self.mcmc_objects[0].algorithm == 'MH':
                print('UQpy: Running Subset Simulation with Metropolis Hastings....')
            elif self.mcmc_objects[0].algorithm == 'MMH':
                print('UQpy: Running Subset Simulation with Modified Metropolis Hastings....')
            elif self.mcmc_objects[0].algorithm == 'DRAM':
                print('UQpy: Running Subset Simulation with DRAM....')
            elif self.mcmc_objects[0].algorithm == 'Stretch':
                print('UQpy: Running Subset Simulation with Stretch....')
            elif self.mcmc_objects[0].algorithm == 'DREAM':
                print('UQpy: Running Subset Simulation with DREAM....')

        [self.pf, self.cov1, self.cov2] = self.run()

        if self.verbose:
            print('UQpy: Subset Simulation Complete!')

#-----------------------------------------------------------------------------------------------------------------------
# The run function executes the chosen subset simulation algorithm
    def run(self):

        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)

        # Generate the initial samples - Level 0
        # Here we need to make sure that we have good initial samples from the target joint density.
        if self.samples_init is None:
            warnings.warn('UQpy: You have not provided initial samples.\n Subset simulation is highly sensitive to the '
                          'initial sample set. It is recommended that the user either:\n'
                          '- Provide an initial set of samples (samples_init) known to follow the distribution; or\n'
                          '- Provide a robust MCMC object that will draw independent initial samples from the '
                          'distribution.')
            self.mcmc_objects[0].run(nsamples=self.nsamples_ss)
            self.samples.append(self.mcmc_objects[0].samples)
        else:
            self.samples.append(self.samples_init)

        # Run the model for the initial samples, sort them by their performance function, and identify the
        # conditional level
        self.runmodel_object.run(samples=np.atleast_2d(self.samples[step]))
        self.g.append(np.asarray(self.runmodel_object.qoi_list))
        g_ind = np.argsort(self.g[step][:, 0])
        self.g_level.append(self.g[step][g_ind[n_keep - 1]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self.cov_sus(step)
        self.d12.append(d1 ** 2)
        self.d22.append(d2 ** 2)

        if self.verbose:
            print('UQpy: Subset Simulation, conditional level 0 complete.')

        while self.g_level[step] > 0 and step < self.max_level:

            # Increment the conditional level
            step = step + 1
            
            # Initialize the samples and the performance function at the next conditional level
            self.samples.append(np.zeros_like(self.samples[step - 1]))
            self.samples[step][:n_keep] = self.samples[step - 1][g_ind[0:n_keep], :]
            self.g.append(np.zeros_like(self.g[step - 1]))
            self.g[step][:n_keep] = self.g[step - 1][g_ind[:n_keep]]

            # Initialize a new MCMC object for each conditional level
            new_mcmc_object = MCMC(dimension=self.mcmc_objects[0].dimension, algorithm=self.mcmc_objects[0].algorithm,
                                   log_pdf_target=self.mcmc_objects[0].log_pdf_target,
                                   seed=np.atleast_2d(self.samples[step][:n_keep, :]))
            self.mcmc_objects.append(new_mcmc_object)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = self.nsamples_ss / self.mcmc_objects[step].nchains
            if n_prop_test.is_integer():
                n_prop = self.nsamples_ss // self.mcmc_objects[step].nchains
            else:
                raise AttributeError(
                    'UQpy: The number of samples per subset (nsamples_ss) must be an integer multiple of '
                    'the number of MCMC chains.')
            
            # Propagate each chain n_prop times and evaluate the model to accept or reject.
            for i in range(n_prop-1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[step].run(nsamples=2*self.mcmc_objects[step].nchains)
                else:
                    self.mcmc_objects[step].run(nsamples=self.mcmc_objects[step].nchains)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[step].samples[i*n_keep:(i+1)*n_keep, :]
                b = self.mcmc_objects[step].samples[(i+1)*n_keep:(i+2)*n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the MCMC state remains unchanged.
                self.samples[step][[x+(i+1)*n_keep for x in ind_true], :] = \
                    self.mcmc_objects[step].samples[ind_true, :]
                self.g[step][[x + (i + 1) * n_keep for x in ind_true], :] = self.g[step][ind_true, :]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[step].samples[[x+(i+1)*n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(g_temp[:, 0] <= self.g_level[step - 1])[0]
                    if x_run.size == 0:
                        print('hmmm')
                    for ii in ind_accept:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = x_run[ii]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = g_temp[ii]

                    # Reject the states with g > g_level
                    ind_reject = np.where(g_temp[:, 0] > self.g_level[step - 1])[0]
                    for ii in ind_reject:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = \
                            self.samples[step][i * n_keep + ind_false[ii]]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = self.g[step][i * n_keep + ind_false[ii]]

            g_ind = np.argsort(self.g[step][:, 0])
            self.g_level.append(self.g[step][g_ind[n_keep]])

            # Estimate coefficient of variation of conditional probability of first level
            d1, d2 = self.cov_sus(step)
            self.d12.append(d1 ** 2)
            self.d22.append(d2 ** 2)

            if self.verbose:
                print('UQpy: Subset Simulation, conditional level ' + step + 'complete.')

        n_fail = len([value for value in self.g[step] if value < 0])

        pf = self.p_cond ** step * n_fail / self.nsamples_ss
        cov1 = np.sqrt(np.sum(self.d12))
        cov2 = np.sqrt(np.sum(self.d22))

        return pf, cov1, cov2

# -----------------------------------------------------------------------------------------------------------------------
# Support functions for subset simulation

    def init_sus(self):
        # Basic error checks for subset simulation.

        # Check that an MCMC object is being passed in.
        if self.mcmc_objects[0] is None:
            raise AttributeError('UQpy: Subset simulation requires the user to pass an MCMC object.')
        if self.runmodel_object is None:
            raise AttributeError(
                'UQpy: No model is defined. Subset simulation requires the user to pass a RunModel '
                'object')

        # Check that a valid conditional probability is specified.
        if type(self.p_cond).__name__ != 'float':
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be of float type.')
        elif self.p_cond <= 0. or self.p_cond >= 1.:
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the number of samples per subset is properly defined.
        if type(self.nsamples_ss).__name__ != 'int':
            raise AttributeError('UQpy: Number of samples per subset (nsamples_ss) must be integer valued.')

        # Check that max_level is an integer
        if type(self.max_level).__name__ != 'int':
            raise AttributeError('UQpy: The maximum subset level (max_level) must be integer valued.')

    def cov_sus(self, step):
    # Compute the coefficient of variation of the samples in a conditional level

        # Here, we assume that the initial samples are drawn to be uncorrelated such that the correction factors do not
        # need to be computed.
        if step == 0:
            d1 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_ss))
            d2 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_ss))

            return d1, d2
        else:
            n_c = int(self.p_cond * self.nsamples_ss)
            n_s = int(1 / self.p_cond)
            indicator = np.reshape(self.g[step] < self.g_level[step], (n_s, n_c))
            gamma = self.corr_factor_gamma(indicator, n_s, n_c)
            g_temp = np.reshape(self.g[step], (n_s, n_c))
            beta_hat = self.corr_factor_beta(g_temp, step)

            d1 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_ss)) * (1 + gamma))
            d2 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_ss)) * (1 + gamma + beta_hat))

            return d1, d2

    # Computes the conventional correlation factor gamma from Au and Beck
    def corr_factor_gamma(self, indicator, n_s, n_c):

        gam = np.zeros(n_s - 1)
        r = np.zeros(n_s)

        ii = indicator * 1
        r_ = ii @ ii.T / n_c - self.p_cond ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_s - 1):
            gam[i] = (1 - ((i + 1) / n_s)) * r[i+1]
        gam = 2 * np.sum(gam)

        return gam

    # Computes the updated correlation factor beta from Shields et al.
    def corr_factor_beta(self, g, step):

        beta = 0
        for i in range(np.shape(g)[1]):
            for j in range(i+1, np.shape(g)[1]):
                if g[0, i] == g[0, j]:
                    beta = beta + 1
        beta = beta*2

        ar = np.asarray(self.mcmc_objects[step].acceptance_rate)
        ar_mean = np.mean(ar)

        factor = 0
        for i in range(np.shape(g)[0]-1):
            factor = factor + (1-(i+1)*np.shape(g)[0]/np.shape(g)[1])*(1-ar_mean)
        factor = factor*2+1

        beta = beta/np.shape(g)[1] * factor
        r_jn = 0

        return beta

########### OLD Subset simulation code #################################################################################

    # dimension=None,
    #
    #
    # pdf_proposal_type=None, pdf_proposal_scale=None,
    # pdf_target=None, log_pdf_target=None, pdf_target_params=None, pdf_target_copula=None,
    # pdf_target_copula_params=None, pdf_target_type='joint_pdf', seed=None,
    # algorithm='MH', jump=1,  nburn=0,
    # model_object=None):
    #
    #
    # # model_script=None, model_object_name=None, input_template=None, var_names=None,
    # # output_script=None, output_object_name=None, n_tasks=1, cores_per_task=1, nodes=1, resume=False,
    # # model_dir=None, cluster=False):

    # self.dimension = dimension
    # self.pdf_proposal_type = pdf_proposal_type
    # self.pdf_proposal_scale = pdf_proposal_scale
    #
    # self.log_pdf_target = log_pdf_target
    # self.pdf_target_copula = pdf_target_copula
    #
    # self.pdf_target_copula_params = pdf_target_copula_params
    # self.jump = jump
    # self.nburn = nburn
    #
    # self.pdf_target_type = pdf_target_type
    # self.pdf_target = pdf_target
    # self.pdf_target_params = pdf_target_params
    # self.algorithm = algorithm
    #
    # if seed is None:
    #     self.seed = np.zeros(self.dimension)
    # else:
    #     self.seed = seed
    # # Hard-wire the maximum number of conditional levels.

    # Select the appropriate Subset Simulation Algorithm
    # if self.mcmc_object.algorithm == 'MMH':
    #     # if self.verbose:
    #     #     print('UQpy: Running Subset Simulation with MMH....')
    #     # [self.pf, self.cov1, self.cov2] = self.run_subsim_mmh()
    #     if self.verbose:
    #         print('UQpy: Running Subset Simulation with Stretch....')
    #     [self.pf, self.cov1, self.cov2] = self.run()
    # elif self.mcmc_object.algorithm == 'Stretch':
    #     if self.verbose:
    #         print('UQpy: Running Subset Simulation with Stretch....')
    #     [self.pf, self.cov1, self.cov2] = self.run()
    # elif self.mcmc_object.algorithm == 'DRAM':
    #     # if self.verbose:
    #     #     print('UQpy: Running Subset Simulation with MMH....')
    #     # [self.pf, self.cov1, self.cov2] = self.run_subsim_mmh()
    #     if self.verbose:
    #         print('UQpy: Running Subset Simulation with Stretch....')
    #     [self.pf, self.cov1, self.cov2] = self.run()
    # # **** Add calls to new methods here.****

    # ------------------------------------------------------------------------------------------------------------------
    # Run Subset Simulation using Modified Metropolis Hastings
    # def run_subsim_mmh(self):
    #     step = 0
    #     n_keep = int(self.p_cond * self.nsamples_ss)
    #
    #     # Generate the initial samples - Level 0
    #     if self.samples_init is None:
    #         self.mcmc_object.run(nsamples=self.nsamples_ss)
    #         self.samples.append(self.mcmc_object.samples)
    #         if self.verbose:
    #             print('UQpy: If the target distribution is other than standard normal, it is highly recommended that '
    #                   'the user provide a set of nsamples_ss samples that follow the target distribution using the '
    #                   'argument samples_init.')
    #     else:
    #         self.samples.append(self.samples_init)
    #
    #     # Run the model for the initial samples,
    #     # sort them by their performance function, and
    #     # identify the conditional level
    #     self.runmodel_object.run(samples=np.atleast_2d(self.samples[step]))
    #     self.g.append(np.asarray(self.runmodel_object.qoi_list))
    #     g_ind = np.argsort(self.g[step][:, 0])
    #     self.g_level.append(self.g[step][g_ind[n_keep-1]])
    #
    #     # Estimate coefficient of variation of conditional probability of first level
    #     d1, d2 = self.cov_sus(step)
    #     self.d12.append(d1 ** 2)
    #     self.d22.append(d2 ** 2)
    #
    #     t = time.time()
    #
    #     if self.verbose:
    #         print('UQpy: Subset Simulation, conditional level 0 complete.')
    #
    #     while self.g_level[step] > 0 and step < self.max_level:
    #
    #         step = step + 1
    #         self.samples.append(np.zeros_like(self.samples[step-1]))
    #         self.samples[step][:n_keep] = self.samples[step - 1][g_ind[0:n_keep], :]
    #         self.g.append(np.zeros_like(self.g[step-1]))
    #         self.g[step][:n_keep] = self.g[step - 1][g_ind[:n_keep]]
    #
    #         for i in range(int(self.nsamples_ss/n_keep)-1):
    #
    #             ind = np.arange(0, n_keep)
    #             # while np.size(ind) != 0:
    #             x_mcmc = np.zeros([np.size(ind), self.samples[step].shape[1]])
    #             x_run = []
    #
    #             k = 0
    #             for j in ind:
    #
    #                 # Generate new candidate states
    #                 self.mcmc_object.samples = None
    #                 self.mcmc_object.seed = np.atleast_2d(self.samples[step][i*n_keep+j, :])
    #                 self.mcmc_object.run(nsamples=1)
    #                 x_mcmc[k] = self.mcmc_object.samples[0, :]
    #
    #                 # Decide whether a new simulation is needed for the proposed state
    #                 if np.array_equal(np.atleast_2d(x_mcmc[k]), self.mcmc_object.seed) is False:
    #                     x_run.append(x_mcmc[k])
    #                 else:
    #                     self.samples[step][(i+1)*n_keep+j] = x_mcmc[k]
    #                     self.g[step][(i+1)*n_keep+j] = self.g[step][i*n_keep+j]
    #
    #                 k += 1
    #
    #             ind = np.where(self.g[step][(i+1)*n_keep:(i+2)*n_keep, 0] == 0)[0]
    #             if np.size(ind) == 0:
    #                 break
    #
    #             # Run the model for the new states.
    #             self.runmodel_object.run(samples=x_run)
    #
    #             # Temporarily save the latest model runs
    #             g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])
    #
    #             # Accept the states with g < g_level
    #             ind_accept = np.where(g_temp[:, 0] <= self.g_level[step - 1])[0]
    #             for ii in ind_accept:
    #                 self.samples[step][(i+1)*n_keep+ind[ii]] = x_mcmc[ind[ii]]
    #                 self.g[step][(i+1)*n_keep+ind[ii]] = g_temp[ii]
    #
    #             ind_reject = np.where(g_temp[:, 0] > self.g_level[step - 1])[0]
    #             for ii in ind_reject:
    #                 self.samples[step][(i+1)*n_keep+ind[ii]] = self.samples[step][i*n_keep+ind[ii]]
    #                 self.g[step][(i+1)*n_keep+ind[ii]] = self.g[step][i*n_keep+ind[ii]]
    #
    #         if self.verbose:
    #             print('UQpy: Subset Simulation, conditional level ' + step + 'complete.')
    #
    #         g_ind = np.argsort(self.g[step][:, 0])
    #         self.g_level.append(self.g[step][g_ind[n_keep]])
    #
    #         # Estimate coefficient of variation of conditional probability of first level
    #         d1, d2 = self.cov_sus(step)
    #         self.d12.append(d1 ** 2)
    #         self.d22.append(d2 ** 2)
    #
    #     n_fail = len([value for value in self.g[step] if value < 0])
    #
    #     pf = self.p_cond ** step * n_fail / self.nsamples_ss
    #     cov1 = np.sqrt(np.sum(self.d12))
    #     cov2 = np.sqrt(np.sum(self.d22))
    #
    #     return pf, cov1, cov2

    #         # Accept or reject each sample
    #
    #
    #     for i in range(int(self.nsamples_ss / n_keep) - 1):
    #
    #         ind = np.arange(0, n_keep)
    #         # while np.size(ind) != 0:
    #         x_mcmc = np.zeros([np.size(ind), self.samples[step].shape[1]])
    #         x_run = []
    #
    #         k = 0
    #         for j in ind:
    #
    #             # Generate new candidate states
    #             ######### Create a new MCMC object for each conditional level. ########
    #
    #             self.mcmc_object.seed = np.atleast_2d(self.samples[step][:n_keep, :])
    #             # self.mcmc_object.samples = self.mcmc_object.seed
    #             self.mcmc_object.nchains = self.mcmc_object.seed.shape[0]
    #             self.mcmc_object.run(nsamples=2)
    #             x_mcmc[k] = self.mcmc_object.samples[0, :]
    #
    #             # Decide whether a new simulation is needed for the proposed state
    #             if np.array_equal(np.atleast_2d(x_mcmc[k]), self.mcmc_object.seed) is False:
    #                 x_run.append(x_mcmc[k])
    #             else:
    #                 self.samples[step][(i + 1) * n_keep + j] = x_mcmc[k]
    #                 self.g[step][(i + 1) * n_keep + j] = self.g[step][i * n_keep + j]
    #
    #             k += 1
    #
    #         ind = np.where(self.g[step][(i + 1) * n_keep:(i + 2) * n_keep, 0] == 0)[0]
    #         if np.size(ind) == 0:
    #             break
    #
    #         # Run the model for the new states.
    #         self.runmodel_object.run(samples=x_run)
    #
    #         # Temporarily save the latest model runs
    #         g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])
    #
    #         # Accept the states with g < g_level
    #         ind_accept = np.where(g_temp[:, 0] <= self.g_level[step - 1])[0]
    #         for ii in ind_accept:
    #             self.samples[step][(i + 1) * n_keep + ind[ii]] = x_mcmc[ind[ii]]
    #             self.g[step][(i + 1) * n_keep + ind[ii]] = g_temp[ii]
    #
    #         ind_reject = np.where(g_temp[:, 0] > self.g_level[step - 1])[0]
    #         for ii in ind_reject:
    #             self.samples[step][(i + 1) * n_keep + ind[ii]] = self.samples[step][i * n_keep + ind[ii]]
    #             self.g[step][(i + 1) * n_keep + ind[ii]] = self.g[step][i * n_keep + ind[ii]]
    #
    #     if self.verbose:
    #         print('UQpy: Subset Simulation, conditional level ' + step + 'complete.')
    #
    #     g_ind = np.argsort(self.g[step][:, 0])
    #     self.g_level.append(self.g[step][g_ind[n_keep]])
    #
    #     # Estimate coefficient of variation of conditional probability of first level
    #     d1, d2 = self.cov_sus(step)
    #     self.d12.append(d1 ** 2)
    #     self.d22.append(d2 ** 2)
    #
    # n_fail = len([value for value in self.g[step] if value < 0])
    #
    # pf = self.p_cond ** step * n_fail / self.nsamples_ss
    # cov1 = np.sqrt(np.sum(self.d12))
    # cov2 = np.sqrt(np.sum(self.d22))
    #
    # return pf, cov1, cov2

    # def run_subsim_stretch(self):


    # Generate the initial samples - Level 0
    # if self.samples_init is None:
    #     x_init = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
    #                   pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
    #                   log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
    #                   pdf_target_copula=self.pdf_target_copula,
    #                   pdf_target_copula_params=self.pdf_target_copula_params,
    #                   pdf_target_type=self.pdf_target_type,
    #                   algorithm='MMH', jump=self.jump, nsamples=self.nsamples_ss, seed=self.seed,
    #                   nburn=self.nburn, verbose=self.verbose)
    #     self.samples.append(x_init.samples)
    # else:
    #     self.samples.append(self.samples_init)

    # g_init = RunModel(samples=self.samples[step], model_script=self.model_script,
    #                   model_object_name=self.model_object_name,
    #                   input_template=self.input_template, var_names=self.var_names,
    #                   output_script=self.output_script,
    #                   output_object_name=self.output_object_name,
    #                   ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes, resume=self.resume,
    #                   verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)

    # self.g.append(np.asarray(g_init.qoi_list))
    # g_ind = np.argsort(self.g[step])
    # self.g_level.append(self.g[step][g_ind[n_keep]])

    # Estimate coefficient of variation of conditional probability of first level
    # d1, d2 = self.cov_sus(step)
    # self.d12.append(d1 ** 2)
    # self.d22.append(d2 ** 2)

    # while self.g_level[step] > 0:
    #
    #     step = step + 1
    #     self.samples.append(self.samples[step - 1][g_ind[0:n_keep]])
    #     self.g.append(self.g[step - 1][g_ind[:n_keep]])
    #
    #     for i in range(self.nsamples_ss - n_keep):
    #
    #         x0 = self.samples[step][i:i+n_keep]
    #
    #         x_mcmc = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
    #                       pdf_proposal_scale=self.pdf_proposal_scale, pdf_target=self.pdf_target,
    #                       log_pdf_target=self.log_pdf_target, pdf_target_params=self.pdf_target_params,
    #                       pdf_target_copula=self.pdf_target_copula,
    #                       pdf_target_copula_params=self.pdf_target_copula_params,
    #                       pdf_target_type=self.pdf_target_type,
    #                       algorithm= self.algorithm, jump=self.jump, nsamples=n_keep+1, seed=x0,
    #                       nburn=self.nburn, verbose=self.verbose)
    #
    #         x_temp = x_mcmc.samples[n_keep].reshape((1, self.dimension))
    #         g_model = RunModel(samples=x_temp, model_script=self.model_script,
    #                            model_object_name=self.model_object_name,
    #                            input_template=self.input_template, var_names=self.var_names,
    #                            output_script=self.output_script,
    #                            output_object_name=self.output_object_name,
    #                            ntasks=self.n_tasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
    #                            resume=self.resume,
    #                            verbose=self.verbose, model_dir=self.model_dir, cluster=self.cluster)
    #
    #         g_temp = g_model.qoi_list
    #
    #         # Accept or reject the sample
    #         if g_temp < self.g_level[step - 1]:
    #             self.samples[step] = np.vstack((self.samples[step], x_temp))
    #             self.g[step] = np.hstack((self.g[step], g_temp[0]))
    #         else:
    #             self.samples[step] = np.vstack((self.samples[step], self.samples[step][i]))
    #             self.g[step] = np.hstack((self.g[step], self.g[step][i]))
    #
    #     g_ind = np.argsort(self.g[step])
    #     self.g_level.append(self.g[step][g_ind[n_keep]])
    #     d1, d2 = self.cov_sus(step)
    #     self.d12.append(d1 ** 2)
    #     self.d22.append(d2 ** 2)
    #
    # n_fail = len([value for value in self.g[step] if value < 0])
    # pf = self.p_cond ** step * n_fail / self.nsamples_ss
    # cov1 = np.sqrt(np.sum(self.d12))
    # cov2 = np.sqrt(np.sum(self.d22))
    #
    # return pf, cov1, cov2


    # -------------------
    # Incomplete Code

    # # Set default dimension to 1
    # if self.dimension is None:
    #     self.dimension = 1
    #
    #
    #
    # # Check that the MCMC algorithm is properly defined.
    # if self.algorithm is None:
    #     self.algorithm = 'MMH'
    # elif self.algorithm not in ['Stretch', 'MMH']:
    #     raise NotImplementedError('Invalid MCMC algorithm. Select from: MMH, Stretch')


    # -------------------

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

    # for i in range(u.size):
    #     ii = indicator[:, g[0, :] == u[i]]
    #     r_jn = r_jn + ii.shape[1]*(ii.shape[1]-1)/2
    #
    # beta = 0

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