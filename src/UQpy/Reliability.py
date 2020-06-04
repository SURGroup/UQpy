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
This module contains functionality for all reliability methods supported in ``UQpy``.
The module currently contains the following classes:

- ``TaylorSeries``: Class to perform reliability analysis using First Order Reliability Method (FORM) and Second Order
  Reliability Method (SORM).
- ``SubsetSimulation``: Class to perform reliability analysis using subset simulation.
"""

import warnings
from inspect import isclass

from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC, MMH
from UQpy.Transformations import *


########################################################################################################################
########################################################################################################################
#                                        Subset Simulation
########################################################################################################################


class SubsetSimulation:
    """
    Perform Subset Simulation to estimate probability of failure.

    This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
    use one of several MCMC algorithms to draw conditional samples.

    **Input:**

    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type `RunModel` (see ``RunModel`` class).

    * **mcmc_class** (Class of type ``SampleMethods.MCMC``)
        Specifies the MCMC algorithm.

        Must be a child class of the ``SampleMethods.MCMC`` parent class. Note: This is `not` and object of the class.
        This input specifies the class itself.

    * **samples_init** (`ndarray`)
        A set of samples from the specified probability distribution. These are the samples from the original
        distribution. They are not conditional samples. The samples must be an array of size
        `nsamples_per_ss x dimension`.

        If `samples_init` is not specified, the Subset_Simulation class will use the `mcmc_class` to draw the initial
        samples.

    * **p_cond** (`float`):
        Conditional probability for each conditional level.

    * **nsamples_per_ss** (`int`)
        Number of samples to draw in each conditional level.

    * **max_level** (`int`)
        Maximum number of allowable conditional levels.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    * **mcmc_kwargs** (`dict`)
        Any additional keyword arguments needed for the specific ``MCMC`` class.

    **Attributes:**

    * **samples** (`list` of `ndarrays`)
         A list of arrays containing the samples in each conditional level.

    * **g** (`list` of `ndarrays`)
        A list of arrays containing the evaluation of the performance function at each sample in each conditional level.

    * **g_level** (`list`)
        Threshold value of the performance function for each conditional level

    * **pf** (`float`)
        Probability of failure estimate

    * **cov1** (`float`)
        Coefficient of variation of the probability of failure estimate assuming independent chains

    * **cov2** (`float`)
        Coefficient of variation of the probability of failure estimate with dependent chains. From [4]_


    **Methods:**
    """

    def __init__(self, runmodel_object, mcmc_class=MMH, samples_init=None, p_cond=0.1, nsamples_per_ss=1000,
                 max_level=10, verbose=False, **mcmc_kwargs):

        # Store the MCMC object to create a new object of this type for each subset
        self.mcmc_kwargs = mcmc_kwargs
        self.mcmc_class = mcmc_class

        # Initialize other attributes
        self.runmodel_object = runmodel_object
        self.samples_init = samples_init
        self.p_cond = p_cond
        self.nsamples_per_ss = nsamples_per_ss
        self.max_level = max_level
        self.verbose = verbose

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')

        self.random_state = self.mcmc_kwargs['random_state']
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        # Perform initial error checks
        self._init_sus()

        # Initialize the mcmc_object from the specified class.
        mcmc_object = self.mcmc_class(**self.mcmc_kwargs)
        self.mcmc_objects = [mcmc_object]

        # Initialize new attributes/variables
        self.samples = list()
        self.g = list()
        self.g_level = list()

        if self.verbose:
            print('UQpy: Running Subset Simulation with MCMC of type: ' + str(type(mcmc_object)))

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

        * **pf** (`float`)
            Probability of failure estimate

        * **cov1** (`float`)
            Coefficient of variation of the probability of failure estimate assuming independent chains

        * **cov2** (`float`)
            Coefficient of variation of the probability of failure estimate with dependent chains. From [4]_

        """

        step = 0
        n_keep = int(self.p_cond * self.nsamples_per_ss)
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
            self.mcmc_objects[0].run(nsamples=self.nsamples_per_ss)
            self.samples.append(self.mcmc_objects[0].samples)
        else:
            self.samples.append(self.samples_init)

        # Run the model for the initial samples, sort them by their performance function, and identify the
        # conditional level
        self.runmodel_object.run(samples=np.atleast_2d(self.samples[step]))
        self.g.append(np.squeeze(self.runmodel_object.qoi_list))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep - 1]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self._cov_sus(step)
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

            # Unpack the attributes

            # Initialize a new MCMC object for each conditional level
            self.mcmc_kwargs['seed'] = np.atleast_2d(self.samples[step][:n_keep, :])
            self.mcmc_kwargs['random_state'] = self.random_state
            new_mcmc_object = self.mcmc_class(**self.mcmc_kwargs)
            self.mcmc_objects.append(new_mcmc_object)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = self.nsamples_per_ss / self.mcmc_objects[step].nchains
            if n_prop_test.is_integer():
                n_prop = self.nsamples_per_ss // self.mcmc_objects[step].nchains
            else:
                raise AttributeError(
                    'UQpy: The number of samples per subset (nsamples_per_ss) must be an integer multiple of '
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
                self.g[step][[x + (i + 1) * n_keep for x in ind_true]] = self.g[step][ind_true]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[step].samples[[x + (i + 1) * n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(g_temp <= self.g_level[step - 1])[0]
                    for ii in ind_accept:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = x_run[ii]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = g_temp[ii]

                    # Reject the states with g > g_level
                    ind_reject = np.where(g_temp > self.g_level[step - 1])[0]
                    for ii in ind_reject:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = \
                            self.samples[step][i * n_keep + ind_false[ii]]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = self.g[step][i * n_keep + ind_false[ii]]

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])

            # Estimate coefficient of variation of conditional probability of first level
            d1, d2 = self._cov_sus(step)
            d12.append(d1 ** 2)
            d22.append(d2 ** 2)

            if self.verbose:
                print('UQpy: Subset Simulation, conditional level ' + str(step) + ' complete.')

        n_fail = len([value for value in self.g[step] if value < 0])

        pf = self.p_cond ** step * n_fail / self.nsamples_per_ss
        cov1 = np.sqrt(np.sum(d12))
        cov2 = np.sqrt(np.sum(d22))

        return pf, cov1, cov2

    # -----------------------------------------------------------------------------------------------------------------------
    # Support functions for subset simulation

    def _init_sus(self):
        """
        Check for errors in the SubsetSimulation class input

        This is an instance method that checks for errors in the input to the SubsetSimulation class. It is
        automatically called when the SubsetSimualtion class is instantiated.

        No inputs or returns.
        """

        # Check that an MCMC class is being passed in.
        if not isclass(self.mcmc_class):
            raise ValueError('UQpy: mcmc_class must be a child class of MCMC. Note it is not an instance of the class.')
        if not issubclass(self.mcmc_class, MCMC):
            raise ValueError('UQpy: mcmc_class must be a child class of MCMC.')

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')

        # Check that a valid conditional probability is specified.
        if type(self.p_cond).__name__ != 'float':
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be of float type.')
        elif self.p_cond <= 0. or self.p_cond >= 1.:
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the number of samples per subset is properly defined.
        if type(self.nsamples_per_ss).__name__ != 'int':
            raise AttributeError('UQpy: Number of samples per subset (nsamples_per_ss) must be integer valued.')

        # Check that max_level is an integer
        if type(self.max_level).__name__ != 'int':
            raise AttributeError('UQpy: The maximum subset level (max_level) must be integer valued.')

    def _cov_sus(self, step):

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
            d1 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss))
            d2 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss))

            return d1, d2
        else:
            n_c = int(self.p_cond * self.nsamples_per_ss)
            n_s = int(1 / self.p_cond)
            indicator = np.reshape(self.g[step] < self.g_level[step], (n_s, n_c))
            gamma = self._corr_factor_gamma(indicator, n_s, n_c)
            g_temp = np.reshape(self.g[step], (n_s, n_c))
            beta_hat = self._corr_factor_beta(g_temp, step)

            d1 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss)) * (1 + gamma))
            d2 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss)) * (1 + gamma + beta_hat))

            return d1, d2

    # Computes the conventional correlation factor gamma from Au and Beck
    def _corr_factor_gamma(self, indicator, n_s, n_c):
        """
        Compute the conventional correlation factor gamma from Au and Beck (Reference [1])

        This is an instance method that computes the correlation factor gamma used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the _cov_sus method.

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
    def _corr_factor_beta(self, g, step):
        """
        Compute the additional correlation factor beta from Shields et al. (Reference [2])

        This is an instance method that computes the correlation factor beta used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the _cov_sus method.

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

        return beta


########################################################################################################################
########################################################################################################################
#                                        First/Second order reliability method
########################################################################################################################
class TaylorSeries:
    """
    Perform First and Second Order Reliability (FORM/SORM) methods.

    This is the parent class to all Taylor series expansion algorithms.

    **Input:**

    * **dist_object** ((list of ) ``Distribution`` object(s)):
        Marginal probability distributions of each random variable. Must be an object of type
        ``DistributionContinuous1D`` or ``JointInd``.

    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type `RunModel` (see ``RunModel`` class).

    * **seed_u** or **seed_x** (`ndarray`):
        The initial starting point for the `Hasofer-Lind` algorithm.

        If `seed_u` is provided, it should be a point in the standard normal space of **U**.

        If `seed_x` is provided, it should be a point in the parameter space of **X**.

        Default: `seed_u = (0, 0, \ldots, 0)`

    * **corr_u** or **corr_x** (`ndarray`):
        Covariance matrix
        If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .

        If `corr_u` is provided, it is the correlation matrix (:math:`\mathbf{C_U}`) of the standard normal random
        vector **U** .

         Default: `cov_u` is specified as the identity matrix.

    * **tol** (`float`):

         Convergence threshold for the `HLRF` algorithm.

         Default: 0.001

    * **n_iter** (`int`):
         Maximum number of iterations for the `HLRF` algorithm.

         Default: 100

    """

    def __init__(self, dist_object, runmodel_object, seed, cov, n_iter, tol):

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], (DistributionContinuous1D, JointInd)):
                    raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')

        if not isinstance(runmodel_object, RunModel):
            raise ValueError('UQpy: A RunModel object is required for the model.')

        self.cov = cov
        self.dimension = self.cov.shape[0]

        self.dist_object = dist_object
        self.n_iter = n_iter
        self.runmodel_object = runmodel_object
        self.tol = tol
        self.seed = seed


class FORM(TaylorSeries):
    """
    A class perform the First Order Reliability Method.

    This is a child class of the ``TaylorSeries`` class.

    **Input:**

    See ``TaylorSeries`` class.

    **Attributes:**

    * **Pf_form** (`float`):
        First-order probability of failure estimate.

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

    * **g_record** (`list`):
        Record of the performance function.

    **Methods:**

    """

    def __init__(self, dist_object, runmodel_object, seed=None, cov=None, n_iter=100,  tol=1e-3):

        super().__init__(dist_object, runmodel_object, seed, cov, n_iter, tol)

        if cov is None:
            if isinstance(dist_object, list):
                cov = np.eye(len(dist_object))
            elif isinstance(dist_object, DistributionContinuous1D):
                cov = np.eye(1)
            elif isinstance(dist_object, JointInd):
                cov = np.eye(len(dist_object.marginals))

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
        self.x = None
        self.alpha = None

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
            self.x = obj.x
            # Jux = obj.Jux
            # Jxu = np.linalg.inv(Jux)

            # 1. evaluate Limit State Function at the point
            self.runmodel_object.run(self.x.reshape(1, -1), append_samples=False)
            qoi = self.runmodel_object.qoi_list[0]
            g_record.append(qoi)

            # 2. evaluate Limit State Function gradient at point u_k and direction cosines
            dg = self.gradient(order='first', point=u,  runmodel_object=self.runmodel_object, cov=self.cov,
                               dist_object=self.dist_object)

            # dg_record.append(np.dot(dg[0, :], Jxu))# use this if the input in gradient function is x
            dg_record.append(dg[0, :])
            norm_grad = np.linalg.norm(dg_record[k])
            self.alpha = - dg_record[k] / norm_grad
            alpha_record.append(self.alpha)

            if k == 0:
                if qoi == 0:
                    g0 = 1
                else:
                    g0 = qoi

            u_check.append(np.linalg.norm(u.reshape(-1, 1) - np.dot(self.alpha.reshape(1, -1), u.reshape(-1, 1))
                                          * self.alpha.reshape(-1, 1)))
            g_check.append(abs(qoi / g0))

            if u_check[k] <= self.tol and g_check[k] <= self.tol:
                conv_flag = 1
            if k == self.n_iter:
                conv_flag = 1

            u_record.append(u)
            x_record.append(self.x)
            if conv_flag == 0:
                direction = (qoi / norm_grad + np.dot(self.alpha.reshape(1, -1), u.reshape(-1, 1))) * \
                            self.alpha.reshape(-1, 1) - u.reshape(-1, 1)
                u_new = (u.reshape(-1, 1) + direction).T
                u = u_new
                k = k + 1

        if k == self.n_iter:
            print('UQpy: Maximum number of iterations {0} was reached before convergence.'.format(self.n_iter))
        else:
            self.HL_beta = np.dot(u, self.alpha.T)
            self.DesignPoint_U = u
            self.DesignPoint_X = self.x
            self.Prob_FORM = stats.norm.cdf(-self.HL_beta)
            self.iterations = k
            self.g_record = g_record
            self.u_record = u_record
            self.x_record = x_record
            self.dg_record = dg_record
            self.alpha_record = alpha_record
            self.u_check = u_check
            self.g_check = g_check

    @staticmethod
    def gradient(dist_object, point, runmodel_object, order='first', cov=None, df_step=0.001, point_qoi=None):
        """
        A method to estimate the derivatives (1st-order, 2nd-order, mixed) of a function using a central difference
        scheme after transformation to the standard normal space.

        This is a static method of the ``FORM`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
            Marginal probability distribution of each random variable. Must be an object of type
            ``DistributionContinuous1D`` or ``JointInd``.

        * **runmodel_object** (``RunModel`` object):
            The computational model. It should be of type ``RunModel`` (see ``RunModel`` class).

        * **corr_u** (`ndarray`):
            Correlation matrix of the standard normal random vector :math:`\mathbf{C_U}`).

        * **point_u** (`ndarray`):
            Point in the uncorrelated standard normal space at which to evaluate the gradient with shape
            `samples.shape=(1, dimension)`

        * **point_qoi** (`float`):
            Value of the model evaluated at point_u. Used only for second derivatives.

        * **order** (`str`):
            Order of the derivative. Available options: 'first', 'second', 'mixed'.

            Default: 'first'.

        * **df_step** (`float`):

            Finite difference step.

            Default: 0.001.

        **Output/Returns:**

        * **du_dj** (`ndarray`):
            Vector of first-order derivatives (if order = 'first').

        * **d2u_dj** (`ndarray`):
            Vector of second-order derivatives (if order = 'second').

        * **d2u_dij** (`ndarray`):
            Vector of mixed derivatives (if order = 'mixed').

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

        if not isinstance(runmodel_object, RunModel) or not callable(runmodel_object):
            raise RuntimeError('UQpy: A RunModel/callable object must be provided as model.')

        def func(m):
            def func_eval(x):
                if callable(m):
                    return m(x)
                elif isinstance(m, RunModel):
                    m.run(samples=x, append_samples=False)
                    return np.array(m.qoi_list)

            return func_eval

        f_eval = func(m=runmodel_object)

        if order.lower() == 'first':
            du_dj = np.zeros([point.shape[0], dimension])

            for ii in range(dimension):
                eps_i = df_step[ii]
                u_i1_j = point.copy()
                u_i1_j[:, ii] = u_i1_j[:, ii] + eps_i
                u_1i_j = point.copy()
                u_1i_j[:, ii] = u_1i_j[:, ii] - eps_i

                obj_plus = Inverse(dist_object=dist_object, samples=u_i1_j, cov=cov)
                temp_x_i1_j = obj_plus.x
                x_i1_j = temp_x_i1_j.reshape(1, -1)
                qoi_plus = f_eval(x_i1_j)

                obj_minus = Inverse(dist_object=dist_object, samples=u_1i_j, cov=cov)
                temp_x_1i_j = obj_minus.x
                x_1i_j = temp_x_1i_j.reshape(1, -1)
                qoi_minus = f_eval(x_1i_j)

                du_dj[:, ii] = ((qoi_plus[0] - qoi_minus[0]) / (2 * eps_i))

            return du_dj

        elif order.lower() == 'second':
            print('Calculating second order derivatives..')
            qoi = kwargs["qoi"]
            d2u_dj = np.zeros([point.shape[0], dimension])
            for ii in range(dimension):
                u_i1_j = point.copy()
                u_i1_j[:, ii] = u_i1_j[:, ii] + df_step[ii]
                u_1i_j = point.copy()
                u_1i_j[:, ii] = u_1i_j[:, ii] - df_step[ii]

                obj = Inverse(dist_object=dist_object, samples=u_i1_j, cov=cov)
                temp_x_i1_j = obj.x
                x_i1_j = temp_x_i1_j.reshape(1, -1)

                obj = Inverse(dist_object=dist_object, samples=u_1i_j, cov=cov)
                temp_x_1i_j = obj.x
                x_1i_j = temp_x_1i_j.reshape(1, -1)

                qoi_plus = f_eval(x_i1_j)
                qoi_minus = f_eval(x_1i_j)

                d2u_dj[:, ii] = ((qoi_plus[0] - 2 * qoi + qoi_minus[0]) / (df_step[ii] * df_step[ii]))

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

                d2u_dij[:, count] = ((qoi_0[0] + qoi_3[0] - qoi_1[0] - qoi_2[0]) / (4 * eps_i1_0 * eps_i1_1))

                count += 1
            return d2u_dij


class SORM(TaylorSeries):
    """
    A class to perform the Second Order Reliability Method.

    ``SORM`` class first performs FORM and then corrects the estimated FORM probability using second-order information.

    ``SORM`` is a child class of the ``TaylorSeries`` class.

    **Input:**

    The ``SORM`` class has the same inputs as the ``TaylorSeries`` class.

    **Output/Returns:**

    The ``SORM`` class has the same outputs as the ``FORM`` class plus

    * **Pf_sorm** (`float`):
        Second-order probability of failure estimate.

    **Methods:**

    """

    def __init__(self, dist_object, runmodel_object, seed=None, cov=None, n_iter=100, tol=1e-3):

        super().__init__(dist_object, runmodel_object, seed, cov, n_iter, tol)

        obj = FORM(dist_object=dist_object, seed=seed, runmodel_object=runmodel_object, cov=cov, n_iter=n_iter, tol=tol)
        self.dimension = obj.dimension
        self.alpha = obj.alpha
        self.DesignPoint_U = obj.DesignPoint_U
        self.model = obj.runmodel_object
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
                                 self.cov, self.dist_object, self.g_record[-1])
        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(self.dg_record[-1])
        kappa = np.linalg.eig(matrix_b[:self.dimension-1, :self.dimension-1])
        self.Prob_SORM = stats.norm.cdf(-self.HL_beta) * np.prod(1 / (1 + self.HL_beta * kappa[0]) ** 0.5)
        self.beta_SORM = -stats.norm.ppf(self.Prob_SORM)

    @staticmethod
    def hessian(point, runmodel_object, cov, dist_obj, qoi, df_step=0.001):
        """
        A function to calculate the hessian matrix  using finite differences. The Hessian matrix is a  square matrix
        of second-order partial derivatives of a scalar-valued function. This is a static method, part of the
        ``Sorm`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
            Marginal probability distribution of each random variable. Must be an object of type
            ``DistributionContinuous1D`` or ``JointInd``.

        * **runmodel_object** (``RunModel`` object):
            The computational model. It should be of type ``RunModel`` (see ``RunModel`` class).

        * **corr_u** (`ndarray`):
            Correlation matrix of the standard normal random vector :math:`\mathbf{C_U}`).

        * **point_u** (`ndarray`):
            Point in the uncorrelated standard normal space at which to evaluate the gradient with shape
            `samples.shape=(1, dimension)`

        * **point_qoi** (`float`):
            Value of the model evaluated at point_u. Used only for second derivatives.

        * **order** (`str`):
            Order of the derivative. Available options: 'first', 'second', 'mixed'.

            Default: 'first'.

        * **df_step** (`float`):
            Finite difference step.

            Default: 0.001.

        **Output/Returns:**

        * **hessian** (`ndarray`):
            The Hessian matrix.

        """
        point = np.atleast_2d(point)
        dimension = point.shape[1]

        dg_second = FORM.gradient(order='second', point=point.reshape(1, -1),
                                  df_step=df_step, runmodel_object=runmodel_object, dist_object=dist_obj,
                                  cov=cov, qoi=qoi)

        dg_mixed = FORM.gradient(order='mixed', point=point.reshape(1, -1),
                                 df_step=df_step, runmodel_object=runmodel_object, dist_object=dist_obj, cov=cov)

        hessian = np.diag(dg_second[0, :])
        import itertools
        range_ = list(range(dimension))
        add_ = 0
        for i in itertools.combinations(range_, 2):
            hessian[i[0], i[1]] = dg_mixed[add_]
            hessian[i[1], i[0]] = hessian[i[0], i[1]]
            add_ += 1

        return hessian
