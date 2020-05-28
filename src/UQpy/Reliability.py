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

"""
This module contains classes and functions for reliability analysis, probability of failure estimation, and rare event
simulation. The module currently contains the following classes:

* SubsetSimulation: Estimate probability of failure using subset simulation.
* TaylorSeries: Estimate probability of failure using FORM or SORM.
"""

from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC
from UQpy.Surrogates import Krig
from UQpy.Transformations import *
import warnings


########################################################################################################################
########################################################################################################################
#                                        Subset Simulation
########################################################################################################################


class SubsetSimulation:
    """
    Perform Subset Simulation to estimate probability of failure.

    This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
    use one of several MCMC algorithms to draw conditional samples.

    **References:**

    1. S.-K. Au and J. L. Beck, “Estimation of small failure probabilities in high dimensions by subset simulation,”
       Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.
    2. Shields, M.D., Giovanis, D.G., and Sundar, V.S. "Subset simulation for problems with strongly non-Gaussian,
       highly anisotropics, and degenerate distributions," Computers & Structures (In Review)

    **Input:**

    :param mcmc_object: An instance of the UQpy.SampleMethods.MCMC class that is used to specify the MCMC algorithm
                        used for conditional sampling

                        This object must be specified.

                        Default: None
    :type mcmc_object: object

    :param runmodel_object: An instance of the UQpy.RunModel.RunModel class that is used to specify the computational
                            model for which probability of failure is to be estimated.

                            This object must be specified.

                            Default: None
    :type runmodel_object: object

    :param samples_init: A set of samples from the specified probability distribution. These are the samples from the
                         original distribution. They are not conditional samples. The samples must be an array of size
                         nsamples_ss x dimension.

                         If samples_init is not specified, the Subset_Simulation class will use the mcmc_object to
                         draw the initial samples.

                         Default: None

    :type samples_init: numpy array

    :param p_cond: Conditional probability for each conditional level

                   Default: 0.1
    :type p_cond: float

    :param nsamples_ss: Number of samples to draw in each conditional level.

                        Default: 1000
    :type nsamples_ss: int

    :param max_level: Maximum number of allowable conditional levels.

                      Default: 10
    :type max_level: int

    :param verbose: Specifies whether algorithm progress is reported in the terminal.

                    Default: False
    :type verbose: boolean

    **Attributes:**

    :param self.samples: A list of arrays containing the samples in each conditional level.
    :type self.samples: list of numpy arrays

    :param self.g: A list of arrays containing the evaluation of the performance function at each sample in each conditional
              level.
    :type self.g: list of numpy arrays

    :param self.g_level: Threshold value of the performance function for each conditional level
    :type self.g_level: list

    :param self.pf: Probability of failure estimate
    :type self.pf: float

    :param self.cov1: Coefficient of variation of the probability of failure estimate assuming independent chains

                      From Reference [1]
    :type self.cov1: float

    :param self.cov2: Coefficient of variation of the probability of failure estimate with dependent chains

                      From Reference [2]
    :type self.cov2: float

    **Authors:**

    Michael D. Shields

    Last Modified: 1/23/20 by Michael D. Shields
    """

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

    # -----------------------------------------------------------------------------------------------------------------------
    # The run function executes the chosen subset simulation algorithm
    def run(self):
        """
        Execute subset simulation

        This is an instance method that runs subset simulation. It is automatically called when the SubsetSimulation
        class is instantiated.

        **Output/Returns:**

        :param pf: Probability of failure estimate
        :type pf: float

        :param cov1: Coefficient of variation of the probability of failure estimate assuming independent chains
        :type cov1: float

        :param cov2: Coefficient of variation of the probability of failure estimate with dependent chains
        :type cov2: float
        """

        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)
        d12 = list()
        d22 = list()

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
        d12.append(d1 ** 2)
        d22.append(d2 ** 2)

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
            new_mcmc_object = MCMC(dimension=self.mcmc_objects[0].dimension,
                                   algorithm=self.mcmc_objects[0].algorithm,
                                   pdf_target=self.mcmc_objects[0].pdf_target,
                                   args_target=self.mcmc_objects[0].args_target,
                                   log_pdf_target=self.mcmc_objects[0].log_pdf_target,
                                   seed=np.atleast_2d(self.samples[step][:n_keep, :]))
            new_mcmc_object.algorithm_inputs = self.mcmc_objects[0].algorithm_inputs
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
            for i in range(n_prop - 1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[step].run(nsamples=2 * self.mcmc_objects[step].nchains)
                else:
                    self.mcmc_objects[step].run(nsamples=self.mcmc_objects[step].nchains)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[step].samples[i * n_keep:(i + 1) * n_keep, :]
                b = self.mcmc_objects[step].samples[(i + 1) * n_keep:(i + 2) * n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the MCMC state remains unchanged.
                self.samples[step][[x + (i + 1) * n_keep for x in ind_true], :] = \
                    self.mcmc_objects[step].samples[ind_true, :]
                self.g[step][[x + (i + 1) * n_keep for x in ind_true], :] = self.g[step][ind_true, :]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[step].samples[[x + (i + 1) * n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(g_temp[:, 0] <= self.g_level[step - 1])[0]
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
            d12.append(d1 ** 2)
            d22.append(d2 ** 2)

            if self.verbose:
                print('UQpy: Subset Simulation, conditional level ' + str(step) + ' complete.')

        n_fail = len([value for value in self.g[step] if value < 0])

        pf = self.p_cond ** step * n_fail / self.nsamples_ss
        cov1 = np.sqrt(np.sum(d12))
        cov2 = np.sqrt(np.sum(d22))

        return pf, cov1, cov2

    # -----------------------------------------------------------------------------------------------------------------------
    # Support functions for subset simulation

    def init_sus(self):
        """
        Check for errors in the SubsetSimulation class input

        This is an instance method that checks for errors in the input to the SubsetSimulation class. It is
        automatically called when the SubsetSimualtion class is instantiated.

        No inputs or returns.
        """

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

        """
        Compute the coefficient of variation of the samples in a conditional level

        This is an instance method that is called after each conditional level is complete to compute the coefficient
        of variation of the conditional probability in that level.

        **Input:**

        :param step: Specifies the conditional level
        :type step: int

        **Output/Returns:**

        :param d1: Coefficient of variation in conditional level assuming independent chains
        :type d1: float

        :param d2: Coefficient of variation in conditional level with dependent chains
        :type d2: float
        """

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
        """
        Compute the conventional correlation factor gamma from Au and Beck (Reference [1])

        This is an instance method that computes the correlation factor gamma used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the cov_sus method.

        **Input:**

        :param indicator: An array of booleans indicating whether the performance function is below the threshold for
                          the conditional probability.
        :type indicator: boolean array

        :param n_s: Number of samples drawn from each Markov chain in each conditional level
        :type n_s: int

        :param n_c: Number of Markov chains in each conditional level
        :type n_c: int

        **Output/Returns:**

        :param gam: Gamma factor in coefficient of variation estimate
        :type gam: float

        """

        gam = np.zeros(n_s - 1)
        r = np.zeros(n_s)

        ii = indicator * 1
        r_ = ii @ ii.T / n_c - self.p_cond ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_s - 1):
            gam[i] = (1 - ((i + 1) / n_s)) * r[i + 1]
        gam = 2 * np.sum(gam)

        return gam

    # Computes the updated correlation factor beta from Shields et al.
    def corr_factor_beta(self, g, step):
        """
        Compute the additional correlation factor beta from Shields et al. (Reference [2])

        This is an instance method that computes the correlation factor beta used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the cov_sus method.

        **Input:**

        :param g: An array containing the performance function evaluation at all points in the current conditional
                  level.
        :type g: numpy array

        :param step: Current conditional level
        :type step: int

        **Output/Returns:**

        :param beta: Beta factor in coefficient of variation estimate
        :type beta: float

        """

        beta = 0
        for i in range(np.shape(g)[1]):
            for j in range(i + 1, np.shape(g)[1]):
                if g[0, i] == g[0, j]:
                    beta = beta + 1
        beta = beta * 2

        ar = np.asarray(self.mcmc_objects[step].acceptance_rate)
        ar_mean = np.mean(ar)

        factor = 0
        for i in range(np.shape(g)[0] - 1):
            factor = factor + (1 - (i + 1) * np.shape(g)[0] / np.shape(g)[1]) * (1 - ar_mean)
        factor = factor * 2 + 1

        beta = beta / np.shape(g)[1] * factor
        r_jn = 0

        return beta


# ######### OLD Subset simulation code #################################################################################

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
    """
    Perform First and Second Order Reliability (FORM/SORM) methods ([1]_, [2]_).

    This is the parent class to all Taylor series expansion algorithms.

    **References:**

    .. [1] R. Rackwitz and R. Fiessler, “Structural reliability under combined random load sequences”,
       Structural Safety, Vol. 22, no. 1, pp: 27–60, 1978.
    .. [2] K. Breitung, “Asymptotic approximations for multinormal integrals”, J. Eng. Mech., ASCE, Vol. 110, no. 3,
       pp: 357–367, 1984.

    **Inputs:**

    * **dist_object** ((list of ) ``Distribution`` object(s)):
                     Probability distribution of each random variable. Must be an object of type
                     ``DistributionContinuous1D`` or ``JointInd``.

    * **model** (Object or a `callable` ):
         The numerical model. It should be of type `RunModel` (see ``RunModel`` class) or ``Krig`` (see ``Surrogates``
         class) object or a `callable`.

    * **seed** (`ndarray`):
         The initial starting point for the `Hasofer-Lind` algorithm. If provided, it should be a point in the parameter
         space **X**. Otherwise, the algorithm starts from point :math:`(0, 0, \ldots, 0)` in the uncorrelated
         standard normal space **U**.

         Default: :math:`(0, 0, \ldots, 0)`

    * **cov** (`ndarray`):
         The distorted correlation  structure (:math:`\mathbf{C_z}`) of the standard normal random vector **Z**. If the
         correlation structure in the parameter space is given (:math:`\mathbf{C_x}`)  then the method
         ``distortion_x_to_z`` from the ``Forward`` class in the ``Nataf`` class should be used  before running FORM to
         obtain :math:`\mathbf{C_z}`.

         Default: The ``identity`` matrix.

    * **cov** (`ndarray`):
         The correlation  structure (:math:`\mathbf{C_X}`) of the random vector **X** .

         Default: The ``identity`` matrix.

    * **tol** (`float`):
         Convergence threshold for the `Hasofer-Lind` algorithm.

         Default: 0.001

    * **n_iter** (`int`):
         Maximum number of iterations for the `Hasofer-Lind` algorithm.

         Default: 100

    """

    def __init__(self, dist_object, model, cov=None, n_iter=100,  tol=1e-3):

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], (DistributionContinuous1D, JointInd)):
                    raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')

        if not isinstance(model, RunModel):
            raise ValueError('UQpy: A RunModel object is required for the model.')


class FORM(TaylorSeries):
    """
    A class perform the First Order reliability method. This is a an child class of the ``TaylorSeries`` class.

    **Inputs:**

    The same as ``TaylorSeries`` class.

    **Attributes:**

    * **Prob_FORM** (`float`):
         First Order probability of failure.

    * **HL_beta** (`float`):
         Hasofer-Lind reliability index.

    * **DesignPoint_U** (`ndarray`):
        Design point in the uncorrelated standard normal space **U**.

    * **DesignPoint_X** (`ndarray`):
        Design point in the parameter space **X**.

    * **alpha** (`ndarray`):
         Direction cosine.

    * **iterations** (`int`):
         Number of model evaluations.

    * **u_record** (`list`):
         Record of all iteration points in the standard normal space **U**.

    * **x_record** (`list`):
         Record of all iteration points in the parameter space **X**.

    * **dg_record** (`list`):
         Record of the model's gradient.

    * **alpha_record** (`list`):
         Record of the alpha (directional cosine).

    * **u_check** (`list`):
         Record of the `u` checks in the standard normal space **U**.

    * **g_check** (`list`):
        Record of the model checks.

    * **g_record** (`list`):
        Record of the performance function.

     **Methods:**

     """

    def __init__(self, dist_object, model, seed=None, cov=None, n_iter=100,  tol=1e-3):

        super().__init__(dist_object, model, cov=None, n_iter=100,  tol=1e-3)

        if cov is None:
            if isinstance(dist_object, list):
                cov = np.eye(len(dist_object))
            elif isinstance(dist_object, DistributionContinuous1D):
                cov = np.eye(1)
            elif isinstance(dist_object, JointInd):
                cov = np.eye(len(dist_object.marginals))
        self.cov = cov
        self.dimension = self.cov.shape[0]

        self.dist_object = dist_object
        self.n_iter = n_iter
        self.model = model
        self.tol = tol
        self.seed = seed

        # Initialize output
        self.HL_beta = None
        self.DesignPoint_U = None
        self.DesignPoint_X = None
        self.alpha = None
        self.Prob_FORM = None
        self.iterations = None
        self.u_record = None
        self.x_record = None
        self.dg_record = None
        self.alpha_record = None
        self.u_check = None
        self.g_check = None
        self.g_record = None

        self._run()

    def _run(self):

        print('UQpy: Running First Order Reliability Method...')

        # initialization
        u_record = list()
        x_record = list()
        g_record = list()
        dg_record = list()
        alpha_record = list()
        g_check = list()
        u_check = list()

        conv_flag = 0
        from UQpy.Transformations import Forward, Inverse
        if self.seed is not None:
            # transform the initial point from the original space x to standard normal space u
            u = Forward(dist_object=self.dist_object, samples=self.seed.reshape(1, -1), cov=self.cov).u
        else:
            u = np.zeros(self.dimension).reshape(1, -1)

        k = 0
        while conv_flag == 0:
            # FORM always starts from the standard normal space
            obj = Inverse(dist_object=self.dist_object, samples=u, cov=self.cov)
            x = obj.x
            # Jux = obj.Jux
            # Jxu = np.linalg.inv(Jux)

            # 1. evaluate Limit State Function at the point
            self.model.run(x.reshape(1, -1), append_samples=False)
            qoi = self.model.qoi_list[0]
            g_record.append(qoi)

            # 2. evaluate Limit State Function gradient at point u_k and direction cosines
            dg = self.gradient(order='first', point=u,  model=self.model, cov=self.cov, dist_object=self.dist_object,
                               run_form=True)

            # dg_record.append(np.dot(dg[0, :], jacobi_x_to_u)) # use this if the input in gradient function is x
            dg_record.append(dg[0, :])
            norm_grad = np.linalg.norm(dg_record[k])
            alpha = - dg_record[k] / norm_grad
            alpha_record.append(alpha)

            if k == 0:
                if qoi == 0:
                    g0 = 1
                else:
                    g0 = qoi

            u_check.append(np.linalg.norm(u.reshape(-1, 1) - np.dot(alpha.reshape(1, -1), u.reshape(-1, 1))
                                          * alpha.reshape(-1, 1)))
            g_check.append(abs(qoi / g0))

            if u_check[k] <= self.tol and g_check[k] <= self.tol:
                conv_flag = 1
            if k == self.n_iter:
                conv_flag = 1

            u_record.append(u)
            x_record.append(x)
            if conv_flag == 0:
                direction = (qoi / norm_grad + np.dot(alpha.reshape(1, -1), u.reshape(-1, 1))) * \
                            alpha.reshape(-1, 1) - u.reshape(-1, 1)
                u_new = (u.reshape(-1, 1) + direction).T
                u = u_new
                k = k + 1

        if k == self.n_iter:
            print('UQpy: Maximum number of iterations {0} was reached before convergence.'.format(self.n_iter))
        else:
            self.HL_beta = np.dot(u, alpha.T)
            self.DesignPoint_U = u
            self.DesignPoint_X = x
            self.Prob_FORM = stats.norm.cdf(-self.HL_beta)
            self.iterations = k
            self.alpha = alpha
            self.g_record = g_record
            self.u_record = u_record
            self.x_record = x_record
            self.dg_record = dg_record
            self.alpha_record = alpha_record
            self.u_check = u_check
            self.g_check = g_check

    @staticmethod
    def gradient(dist_object, point, model, order='first', cov=None, df_step=0.001, run_form=False, read_qoi=None):
        """
        A method to estimate the gradients (1st, 2nd, mixed) of a function using a finite difference scheme. First
        order gradients are calculated using forward finite differences. This is a static method, part of the
        ``Form`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **model** (Object or a `callable` ):
            The numerical model. It should be of type `RunModel` (see ``RunModel`` class) or ``Krig`` (see ``Surrogates``
            class) object or a `callable`.

        * **run_form** (Boolean):
            If the ``gradient`` method is used in the framework of FORM, then the input point is in the standard
            normal space **U**. In this case, the ``Nataf`` class is used to transform the point in the parameter
            space so it can be used with the model.

            Default: False

        * **cov** (`ndarray`):
            The correlation  structure (:math:`\mathbf{C_X}`) of the random vector **X** (optional).

        * **point** (`ndarray`):
            The point to evaluate the gradient with shape ``samples.shape=(1, dimension)

        * **order** (`str`):
            Order of the gradient. Available options: 'first', 'second', 'mixed'.

            Default: 'First'.

        * **df_step** (`float`):
            Finite difference step.

            Default: 0.001.

        * **read_qoi** (`float`):
            Value of the model provided manually.

        **Output/Returns:**

        * **du_dj** (`ndarray`):
            Vector of first-order gradients (if order = 'first').

        * **d2u_dj** (`ndarray`):
            Vector of second-order gradients (if order = 'second').

        * **d2u_dij** (`ndarray`):
            Vector of mixed gradients (if order = 'mixed').

        """

        point = np.atleast_2d(point)
        dimension = point.shape[1]

        if dimension is None:
            raise ValueError('Error: Dimension must be defined')

        if isinstance(df_step, float):
            df_step = [df_step] * dimension
        elif isinstance(df_step, list):
            if len(df_step) != 1 and len(df_step) != dimension:
                raise ValueError('UQpy: Inconsistent dimensions.')
            if len(df_step) == 1:
                df_step = [df_step[0]] * dimension

        if isinstance(model, Krig):
            qoi = model.interpolate(samples)
        elif isinstance(model, RunModel) and read_qoi is None:
            qoi = model.qoi_list[0]
        elif isinstance(model, RunModel) and read_qoi is not None:
            qoi = read_qoi
        elif isinstance(model, callable):
            qoi = model(samples)
        else:
            raise RuntimeError('UQpy: A RunModel/Krig/callable object must be provided as model.')

        def func(m):
            def func_eval(x):
                if isinstance(m, Krig):
                    return m.interpolate(x=x)
                elif isinstance(m, RunModel):
                    m.run(samples=x, append_samples=False)
                    return np.array(m.qoi_list)
                else:
                    return m(x)

            return func_eval

        f_eval = func(m=model)

        if order.lower() == 'first':
            du_dj = np.zeros([point.shape[0], dimension])

            for ii in range(dimension):
                eps_i = df_step[ii]
                u_i1_j = point.copy()
                u_i1_j[:, ii] = u_i1_j[:, ii] + eps_i

                if run_form is True:
                    obj = Inverse(dist_object=dist_object, samples=u_i1_j, cov=cov)
                    temp_x_i1_j = obj.x
                    x_i1_j = temp_x_i1_j.reshape(1, -1)

                    qoi_plus = f_eval(x_i1_j)
                else:
                    qoi_plus = f_eval(u_i1_j)

                du_dj[:, ii] = ((qoi_plus[0] - qoi) / eps_i)

            return du_dj

        elif order.lower() == 'second':
            print('Calculating second order derivatives..')
            d2u_dj = np.zeros([point.shape[0], dimension])
            for ii in range(dimension):
                u_i1_j = point.copy()
                u_i1_j[:, ii] = u_i1_j[:, ii] + df_step[ii]
                u_1i_j = point.copy()
                u_1i_j[:, ii] = u_1i_j[:, ii] - df_step[ii]

                if run_form is True:
                    obj = Inverse(dist_object=dist_object, samples=u_i1_j, cov=cov)
                    temp_x_i1_j = obj.x
                    x_i1_j = temp_x_i1_j.reshape(1, -1)

                    obj = Inverse(dist_object=dist_object, samples=u_1i_j, cov=cov)
                    temp_x_1i_j = obj.x
                    x_1i_j = temp_x_1i_j.reshape(1, -1)

                    qoi_plus = f_eval(x_i1_j)
                    qoi_minus = f_eval(x_1i_j)
                else:
                    qoi_plus = f_eval(u_i1_j)
                    qoi_minus = f_eval(u_1i_j)
                d2u_dj[:, ii] = ((qoi_plus[0] - 2 * qoi + qoi_minus[0]) / (df_step[ii] *df_step[ii]))

            return d2u_dj

        elif order.lower() == 'mixed':

            import itertools
            range_ = list(range(dimension))
            d2u_dij = np.zeros([point.shape[0], int(dimension * (dimension - 1) / 2)])
            count = 0
            for i in itertools.combinations(range_, 2):
                u_i1_j1 = point.copy()
                u_i1_1j = point.copy()
                u_1i_j1 = point.copy()
                u_1i_1j = point.copy()

                eps_i1_0 = df_step[i[0]]
                eps_i1_1 = df_step[i[1]]

                u_i1_j1[:, i[0]] += eps_i1_0
                u_i1_j1[:, i[1]] += eps_i1_1

                u_i1_1j[:, i[0]] += eps_i1_0
                u_i1_1j[:, i[1]] -= eps_i1_1

                u_1i_j1[:, i[0]] -= eps_i1_0
                u_1i_j1[:, i[1]] += eps_i1_1

                u_1i_1j[:, i[0]] -= eps_i1_0
                u_1i_1j[:, i[1]] -= eps_i1_1

                if run_form:
                    obj = Inverse(dist_object=dist_object, samples=u_i1_j1, cov=cov)
                    temp_x_i1_j1 = obj.x
                    x_i1_j1 = temp_x_i1_j1.reshape(1, -1)

                    obj = Inverse(dist_object=dist_object, samples=u_i1_1j, cov=cov)
                    temp_x_i1_1j = obj.x
                    x_i1_1j = temp_x_i1_1j[0].reshape(1, -1)

                    obj = Inverse(dist_object=dist_object, samples=u_1i_j1, cov=cov)
                    temp_x_1i_j1 = obj.x
                    x_1i_j1 = temp_x_1i_j1[0].reshape(1, -1)

                    obj = Inverse(dist_object=dist_object, samples=u_1i_1j, cov=cov)
                    temp_x_1i_1j = obj.x
                    x_1i_1j = temp_x_1i_1j.reshape(1, -1)

                    qoi_0 = f_eval(x_i1_j1)
                    qoi_1 = f_eval(x_i1_1j)
                    qoi_2 = f_eval(x_1i_j1)
                    qoi_3 = f_eval(x_1i_1j)
                else:
                    qoi_0 = f_eval(u_i1_j1)
                    qoi_1 = f_eval(u_i1_1j)
                    qoi_2 = f_eval(u_1i_j1)
                    qoi_3 = f_eval(u_1i_1j)

                d2u_dij[:, count] = ((qoi_0[0] + qoi_3[0] - qoi_1[0] - qoi_2[0]) / (4 * eps_i1_0 * eps_i1_1))

                count += 1
            return d2u_dij


class SORM(TaylorSeries):
    """
    A class perform the Second Order reliability method. ``Sorm`` class performs  FORM and then corrects the estimated
    FORM probability using second-order information. This is a an child class of the ``TaylorSeries`` class.

    **Inputs:**

    The ``Sorm`` class has the same inputs with the ``TaylorSeries`` class.

    **Output/Returns:**

    The ``Sorm`` class has the same outputs with the ``Form`` class plus

    * **Prob_FORM** (`float`):
        Second Order probability of failure.

     **Methods:**

    """

    def __init__(self, dist_object, model, seed=None, cov=None, n_iter=100, tol=1e-3):

        super().__init__(dist_object, model, cov=None, n_iter=100, tol=1e-3)

        obj = FORM(dist_object=dist_object, seed=seed, model=model, cov=cov, n_iter=n_iter, tol=tol)
        self.dimension = obj.dimension
        self.alpha = obj.alpha
        self.DesignPoint_U = obj.DesignPoint_U
        self.model = obj.model
        self.cov = obj.cov
        self.dist_object = dist_object
        self.dg_record = obj.dg_record
        self.g_record = obj.g_record
        self.HL_beta = obj.HL_beta
        self.Prob_FORM = obj.Prob_FORM

        print('UQpy: Running SORM...')

        matrix_a = np.fliplr(np.eye(self.dimension))
        matrix_a[:, 0] = self.alpha

        def normalize(v):
            return v / np.sqrt(v.dot(v))

        q = np.zeros(shape=(self.dimension, self.dimension))
        q[:, 0] = normalize(matrix_a[:, 0])

        for i in range(1, self.dimension):
            ai = matrix_a[:, i]
            for j in range(0, i):
                aj = matrix_a[:, j]
                t = ai.dot(aj)
                ai = ai - t * aj
            q[:, i] = normalize(ai)

        r1 = np.fliplr(q).T
        hessian_g = self.hessian(self.DesignPoint_U, self.model,
                                 self.cov, self.dist_object, self.g_record[-1], run_form=True)
        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(self.dg_record[-1])
        kappa = np.linalg.eig(matrix_b[:self.dimension-1, :self.dimension-1])
        self.Prob_SORM = stats.norm.cdf(-self.HL_beta) * np.prod(1 / (1 + self.HL_beta * kappa[0]) ** 0.5)
        self.beta_SORM = -stats.norm.ppf(self.Prob_SORM)

    @staticmethod
    def hessian(point, model, cov, dist_obj, read_qoi, df_step=0.001, run_form=False):
        """
        A function to calculate the hessian matrix  using finite differences. The Hessian matrix is a  square matrix
        of second-order partial derivatives of a scalar-valued function. This is a static method, part of the
        ``Sorm`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **model** (Object or a `callable` ):
            The numerical model. It should be of type `RunModel` (see ``RunModel`` class) or ``Krig`` (see ``Surrogates``
            class) object or a `callable`.

        * **run_form** (Boolean):
            If the ``gradient`` method is used in the framework of FORM, then the input point is in the standard
            normal space **U**. In this case, the ``Nataf`` class is used to transform the point in the parameter
            space so it can be used with the model.

            Default: False

        * **cov** (`ndarray`):
            The correlation  structure (:math:`\mathbf{C_X}`) of the random vector **X** (optional).

        * **point** (`ndarray`):
            The point to evaluate the gradient with shape ``samples.shape=(1, dimension)

        * **read_qoi** (`float`):
            Value of the model provided manually..

        * **df_step** (`float`):
            Finite difference step.

            Default: 0.001.

        **Output/Returns:**

        * **hessian** (`ndarray`):
            The hessian matrix.

        """
        point = np.atleast_2d(point)
        dimension = point.shape[1]

        dg_second = FORM.gradient(order='second', point=point.reshape(1, -1),
                                  df_step=df_step, model=model, dist_object=dist_obj,
                                  cov=cov, read_qoi=read_qoi, run_form=run_form)

        dg_mixed = FORM.gradient(order='mixed', point=point.reshape(1, -1),
                                 df_step=df_step, model=model, dist_object=dist_obj,
                                 read_qoi=read_qoi, cov=cov, run_form=run_form)

        hessian = np.diag(dg_second[0, :])
        import itertools
        range_ = list(range(dimension))
        add_ = 0
        for i in itertools.combinations(range_, 2):
            hessian[i[0], i[1]] = dg_mixed[add_]
            hessian[i[1], i[0]] = hessian[i[0], i[1]]
            add_ += 1

        return hessian
