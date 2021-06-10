import numpy as np
import scipy.stats as stats

from UQpy.sample_methods.LatinHypercubeSampling import LatinHypercubeSampling


class AdaptiveKrigingMonteCarlo:
    """
    Adaptively sample for construction of a kriging surrogate for different objectives including reliability,
    optimization, and global fit.


    **Inputs:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **runmodel_object** (``RunModel`` object):
        A ``RunModel`` object, which is used to evaluate the model.

    * **samples** (`ndarray`):
        The initial samples at which to evaluate the model.

        Either `samples` or `nstart` must be provided.

    * **krig_object** (`class` object):
        A kriging surrogate model, this object must have ``fit`` and ``predict`` methods.

        May be an object of the ``UQpy`` ``kriging`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

    * **nsamples** (`int`):
        Total number of samples to be drawn (including the initial samples).

        If `nsamples` and `samples` are provided when instantiating the class, the ``run`` method will automatically be
        called. If either `nsamples` or `samples` is not provided, ``AKMCS`` can be executed by invoking the ``run``
        method and passing `nsamples`.

    * **nlearn** (`int`):
        Number of samples generated for evaluation of the learning function. Samples for the learning set are drawn
        using ``LHS``.

    * **qoi_name** (`dict`):
        Name of the quantity of interest. If the quantity of interest is a dictionary, this is used to convert it to
        a list

    * **learning_function** (`str` or `function`):
        Learning function used as the selection criteria to identify new samples.

        Built-in options:
                    1. 'U' - U-function \n
                    2. 'EFF' - Expected Feasibility Function \n
                    3. 'Weighted-U' - Weighted-U function \n
                    4. 'EIF' - Expected Improvement Function \n
                    5. 'EGIF' - Expected Global Improvement Fit \n

        `learning_function` may also be passed as a user-defined callable function. This function must accept a kriging
        surrogate model object with ``fit`` and ``predict`` methods, the set of learning points at which to evaluate the
        learning function, and it may also take an arbitrary number of additional parameters that are passed to
        ``AKMCS`` as `**kwargs`.

    * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False.

    * **kwargs**
        Used to pass parameters to `learning_function`.

        For built-in `learning_functions`, see the requisite inputs in the method list below.

        For user-defined `learning_functions`, these will be defined by the requisite inputs to the user-defined method.


    **Attributes:**

    * **samples** (`ndarray`):
        `ndarray` containing the samples at which the model is evaluated.

    * **lf_values** (`list`)
        The learning function evaluated at new sample points.


    **Methods:**

    """

    def __init__(self, distributions, runmodel_object, kriging, samples=None, samples_number=None,
                 learning_samples_number=None, qoi_name=None, learning_function='U', n_add=1, random_state=None,
                 verbose=False, **kwargs):

        # Initialize the internal variables of the class.
        self.runmodel_object = runmodel_object
        self.samples = np.array(samples)
        self.learning_samples_number = learning_samples_number
        self.initial_samples_number = None
        self.verbose = verbose
        self.qoi_name = qoi_name

        self.learning_function = learning_function
        self.learning_set = None
        self.dist_object = distributions
        self.samples_number = samples_number

        self.moments = None
        self.n_add = n_add
        self.indicator = False
        self.pf = []
        self.cov_pf = []
        self.dimension = 0
        self.qoi = None
        self.krig_model = None
        self.kwargs = kwargs

        # Initialize and run preliminary error checks.
        self.dimension = len(distributions)

        if samples is not None:
            if self.dimension != self.samples.shape[1]:
                raise NotImplementedError("UQpy Error: Dimension of samples and distribution are inconsistent.")

        if self.learning_function not in ['EFF', 'U', 'Weighted-U', 'EIF', 'EIGF']:
            raise NotImplementedError("UQpy Error: The provided learning function is not recognized.")
        elif self.learning_function == 'EIGF':
            self.learning_function = self.expected_improvement_global_fit
        elif self.learning_function == 'EIF':
            if 'eif_stop' not in self.kwargs:
                self.kwargs['eif_stop'] = 0.01
            self.learning_function = self.expected_improvement_function
        elif self.learning_function == 'U':
            if 'u_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 2
            self.learning_function = self.u
        elif self.learning_function == 'Weighted-U':
            if 'u_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 2
            self.learning_function = self.weighted_u
        else:
            if 'a' not in self.kwargs:
                self.kwargs['a'] = 0
            if 'epsilon' not in self.kwargs:
                self.kwargs['epsilon'] = 2
            if 'eff_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 0.001
            self.learning_function = self.expected_feasiblity_function

        from UQpy.distributions import DistributionContinuous1D, JointIndependent

        if isinstance(distributions, list):
            for i in range(len(distributions)):
                if not isinstance(distributions[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            if not isinstance(distributions, (DistributionContinuous1D, JointIndependent)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if hasattr(kriging, 'fit') and hasattr(kriging, 'predict'):
            self.krig_object = kriging
        else:
            raise NotImplementedError("UQpy: krig_object must have 'fit' and 'predict' methods.")

        if self.verbose:
            print('UQpy: AKMCS - Running the initial sample set using RunModel.')

        # Evaluate model at the training points
        if len(self.runmodel_object.qoi_list) == 0 and samples is not None:
            self.runmodel_object.run(samples=self.samples, append_samples=False)
        if samples is not None:
            if len(self.runmodel_object.qoi_list) != self.samples.shape[0]:
                raise NotImplementedError("UQpy: There should be no model evaluation or Number of samples and model "
                                          "evaluation in RunModel object should be same.")

        if self.samples_number is not None:
            if self.samples_number <= 0 or type(self.samples_number).__name__ != 'int':
                raise NotImplementedError("UQpy: Number of samples to be generated 'nsamples' should be a positive "
                                          "integer.")

            if samples is not None:
                self.run(samples_number=self.samples_number)

    def run(self, samples_number, samples=None, append_samples=True, initial_samples_number=None):
        """
        Execute the ``AKMCS`` learning iterations.

        The ``run`` method is the function that performs iterations in the ``AKMCS`` class. If `nsamples` is
        provided when defining the ``AKMCS`` object, the ``run`` method is automatically called. The user may also
        call the ``run`` method directly to generate samples. The ``run`` method of the ``AKMCS`` class can be invoked
        many times.

        **Inputs:**

        * **nsamples** (`int`):
            Total number of samples to be drawn (including the initial samples).

        * **samples** (`ndarray`):
            Samples at which to evaluate the model.

        * **nstart** (`int`):
            Number of initial samples, randomly generated using ``LHS`` class.

            Either `samples` or `nstart` must be provided.

        * **append_samples** (`boolean`)
            Append new samples and model evaluations to the existing samples and model evaluations.

            If ``append_samples = False``, all previous samples and the corresponding quantities of interest from their
            model evaluations are deleted.

            If ``append_samples = True``, samples and their resulting quantities of interest are appended to the
            existing ones.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the
        ``AKMCS`` class.

        """

        self.samples_number = samples_number
        self.initial_samples_number = initial_samples_number

        if samples is not None:
            # New samples are appended to existing samples, if append_samples is TRUE
            if append_samples:
                if len(self.samples.shape) == 0:
                    self.samples = np.array(samples)
                else:
                    self.samples = np.vstack([self.samples, np.array(samples)])
            else:
                self.samples = np.array(samples)
                self.runmodel_object.qoi_list = []

            if self.verbose:
                print('UQpy: AKMCS - Evaluating the model at the sample set using RunModel.')

            self.runmodel_object.run(samples=samples, append_samples=append_samples)
        else:
            if len(self.samples.shape) == 0:
                if self.initial_samples_number is None:
                    raise NotImplementedError("UQpy: User should provide either 'samples' or 'nstart' value.")
                if self.verbose:
                    print('UQpy: AKMCS - Generating the initial sample set using Latin hypercube sampling.')
                self.samples = LatinHypercubeSampling(distributions=self.dist_object, samples_number=self.initial_samples_number,
                                                      random_state=self.random_state).samples
                self.runmodel_object.run(samples=self.samples)

        if self.verbose:
            print('UQpy: Performing AK-MCS design...')

        # If the quantity of interest is a dictionary, convert it to a list
        self.qoi = [None] * len(self.runmodel_object.qoi_list)
        if type(self.runmodel_object.qoi_list[0]) is dict:
            for j in range(len(self.runmodel_object.qoi_list)):
                self.qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
        else:
            self.qoi = self.runmodel_object.qoi_list

        # Train the initial kriging model.
        self.krig_object.fit(self.samples, self.qoi)
        self.krig_model = self.krig_object.predict

        # kwargs = {"n_add": self.n_add, "parameters": self.kwargs, "samples": self.samples, "qoi": self.qoi,
        #           "dist_object": self.dist_object}

        # ---------------------------------------------
        # Primary loop for learning and adding samples.
        # ---------------------------------------------

        for i in range(self.samples.shape[0], self.samples_number):
            # Initialize the population of samples at which to evaluate the learning function and from which to draw
            # in the sampling.

            lhs = LatinHypercubeSampling(distributions=self.dist_object, samples_number=self.learning_samples_number, random_state=self.random_state)
            self.learning_set = lhs.samples.copy()

            # Find all of the points in the population that have not already been integrated into the training set
            rest_pop = np.array([x for x in self.learning_set.tolist() if x not in self.samples.tolist()])

            # Apply the learning function to identify the new point to run the model.

            # new_point, lf, ind = self.learning_function(self.krig_model, rest_pop, **kwargs)
            new_point, lf, ind = self.learning_function(self.krig_model, rest_pop, n_add=self.n_add,
                                                        parameters=self.kwargs, samples=self.samples, qoi=self.qoi,
                                                        dist_object=self.dist_object)

            # Add the new points to the training set and to the sample set.
            self.samples = np.vstack([self.samples, np.atleast_2d(new_point)])

            # Run the model at the new points
            self.runmodel_object.run(samples=new_point, append_samples=True)

            # If the quantity of interest is a dictionary, convert it to a list
            self.qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    self.qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                self.qoi = self.runmodel_object.qoi_list

            # Retrain the surrogate model
            self.krig_object.fit(self.samples, self.qoi, optimizations_number=1)
            self.krig_model = self.krig_object.predict

            # Exit the loop, if error criteria is satisfied
            if ind:
                print("UQpy: Learning stops at iteration: ", i)
                break

            if self.verbose:
                print("Iteration:", i)

        if self.verbose:
            print('UQpy: AKMCS complete')

    # ------------------
    # LEARNING FUNCTIONS
    # ------------------
    @staticmethod
    def expected_improvement_global_fit(surr, pop, **kwargs):
        """
        Expected Improvement for Global Fit (EIGF) learning function. See [7]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EIGF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. For
            ``EIGF``, this dictionary is empty as no stopping criterion is specified.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eigf_lf** (`ndarray`)
            EIGF learning function evaluated at the new sample points.

        """
        samples = kwargs['samples']
        qoi = kwargs['qoi']
        n_add = kwargs['n_add']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(samples))
        neighbors = knn.kneighbors(np.atleast_2d(pop), return_distance=False)

        # noinspection PyTypeChecker
        qoi_array = np.array([qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(g - qoi_array) + np.square(sig)
        rows = u[:, 0].argsort()[(np.size(g) - n_add):]

        indicator = False
        new_samples = pop[rows, :]
        eigf_lf = u[rows, :]

        return new_samples, eigf_lf, indicator

    @staticmethod
    def u(surr, pop, **kwargs):
        """
        U-function for reliability analysis. See [3] for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the U-function is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `u_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **u_lf** (`ndarray`)
            U learning function evaluated at the new sample points.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:n_add]

        indicator = False
        if min(u[:, 0]) >= parameters['u_stop']:
            indicator = True

        new_samples = pop[rows, :]
        u_lf = u[rows, 0]
        return new_samples, u_lf, indicator

    @staticmethod
    def weighted_u(surr, pop, **kwargs):
        """
        Probability Weighted U-function for reliability analysis. See [5]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the weighted U-function is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `u_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.

        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **w_lf** (`ndarray`)
            Weighted U learning function evaluated at the new sample points.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']
        samples = kwargs['samples']
        dist_object = kwargs['dist_object']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        u = abs(g) / sig
        p1, p2 = np.ones([pop.shape[0], pop.shape[1]]), np.ones([samples.shape[0], pop.shape[1]])
        for j in range(samples.shape[1]):
            p1[:, j] = dist_object[j].pdf(np.atleast_2d(pop[:, j]).T)
            p2[:, j] = dist_object[j].pdf(np.atleast_2d(samples[:, j]).T)

        p1 = p1.prod(1).reshape(u.size, 1)
        max_p = max(p2.prod(1))
        u_ = u * ((max_p - p1) / max_p)
        rows = u_[:, 0].argsort()[:n_add]

        indicator = False
        if min(u[:, 0]) >= parameters['weighted_u_stop']:
            indicator = True

        new_samples = pop[rows, :]
        w_lf = u_[rows, :]
        return new_samples, w_lf, indicator

    @staticmethod
    def expected_feasiblity_function(surr, pop, **kwargs):
        """
        Expected Feasibility Function (EFF) for reliability analysis, see [6]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EFF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            these include `a`, `epsilon`, and `eff_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eff_lf** (`ndarray`)
            EFF learning function evaluated at the new sample points.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])
        # reliability threshold: a_ = 0
        # EGRA method: epsilon = 2*sigma(x)
        a_, ep = parameters['eff_a'], parameters['eff_epsilon']*sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (2 * stats.norm.cdf(t1) - stats.norm.cdf(t2) - stats.norm.cdf(t3))
        eff += -sig * (2 * stats.norm.pdf(t1) - stats.norm.pdf(t2) - stats.norm.pdf(t3))
        eff += ep * (stats.norm.cdf(t3) - stats.norm.cdf(t2))
        rows = eff[:, 0].argsort()[-n_add:]

        indicator = False
        if max(eff[:, 0]) <= parameters['eff_stop']:
            indicator = True

        new_samples = pop[rows, :]
        eff_lf = eff[rows, :]
        return new_samples, eff_lf, indicator

    @staticmethod
    def expected_improvement_function(surr, pop, **kwargs):
        """
        Expected Improvement Function (EIF) for Efficient Global Optimization (EFO). See [4]_ for a detailed
        explanation.


        **Inputs:**

        * **surr** (`class` object):
            A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EIF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `eif_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eif_lf** (`ndarray`)
            EIF learning function evaluated at the new sample points.
        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']
        qoi = kwargs['qoi']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        fm = min(qoi)
        eif = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = eif[:, 0].argsort()[(np.size(g) - n_add):]

        indicator = False
        if max(eif[:, 0]) / abs(fm) <= parameters['eif_stop']:
            indicator = True

        new_samples = pop[rows, :]
        eif_lf = eif[rows, :]
        return new_samples, eif_lf, indicator
