import numpy as np
from UQpy.sampling.LatinHypercubeSampling import LatinHypercubeSampling
from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import LearningFunction
from UQpy.distributions import DistributionContinuous1D, JointIndependent
from UQpy.sampling.latin_hypercube_criteria import Random

class AdaptiveKriging:
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

    def __init__(self, distributions, runmodel_object, surrogate, samples=None, samples_number=None,
                 learning_samples_number=None, qoi_name=None, learning_function='U', n_add=1, random_state=None,
                 verbose=False):

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
        self.prediction_model = None

        # Initialize and run preliminary error checks.
        self.dimension = len(distributions)

        if samples is not None:
            if self.dimension != self.samples.shape[1]:
                raise NotImplementedError("UQpy Error: Dimension of samples and distribution are inconsistent.")

        if not isinstance(self.learning_function, LearningFunction):
            raise NotImplementedError("UQpy Error: The provided learning function is not recognized.")

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

        if hasattr(surrogate, 'fit') and hasattr(surrogate, 'predict'):
            self.surrogate = surrogate
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


                random_criterion = Random(random_state=self.random_state)
                latin_hypercube_sampling = \
                    LatinHypercubeSampling(distributions=self.dist_object, samples_number=2,
                                           verbose=True, criterion=random_criterion)
                self.samples = latin_hypercube_sampling.samples
                self.runmodel_object.run(samples=self.samples)

        if self.verbose:
            print('UQpy: Performing AK-MCS design...')

        # If the quantity of interest is a dictionary, convert it to a list
        self._convert_qoi_tolist()

        # Train the initial kriging model.
        self.surrogate.fit(self.samples, self.qoi)
        self.prediction_model = self.surrogate.predict

        # ---------------------------------------------
        # Primary loop for learning and adding samples.
        # ---------------------------------------------

        for i in range(self.samples.shape[0], self.samples_number):
            # Initialize the population of samples at which to evaluate the learning function and from which to draw
            # in the sampling.
            random_criterion = Random(random_state=self.random_state)
            lhs = \
                LatinHypercubeSampling(distributions=self.dist_object, samples_number=self.learning_samples_number,
                                       verbose=True, criterion=random_criterion)

            self.learning_set = lhs.samples.copy()

            # Find all of the points in the population that have not already been integrated into the training set
            rest_pop = np.array([x for x in self.learning_set.tolist() if x not in self.samples.tolist()])

            # Apply the learning function to identify the new point to run the model.

            # new_point, lf, ind = self.learning_function(self.krig_model, rest_pop, **kwargs)
            new_point, lf, ind = self.learning_function\
                .evaluate_function(distributions=self.dist_object,
                                   n_add=self.n_add,
                                   surrogate=self.surrogate,
                                   population=rest_pop,
                                   qoi=self.qoi,
                                   samples=self.samples)


            # Add the new points to the training set and to the sample set.
            self.samples = np.vstack([self.samples, np.atleast_2d(new_point)])

            # Run the model at the new points
            self.runmodel_object.run(samples=new_point, append_samples=True)

            # If the quantity of interest is a dictionary, convert it to a list
            self._convert_qoi_tolist()

            # Retrain the surrogate model
            self.surrogate.fit(self.samples, self.qoi, optimizations_number=1)
            self.prediction_model = self.surrogate.predict

            # Exit the loop, if error criteria is satisfied
            if ind:
                print("UQpy: Learning stops at iteration: ", i)
                break

            if self.verbose:
                print("Iteration:", i)

        if self.verbose:
            print('UQpy: AKMCS complete')

    def _convert_qoi_tolist(self):
        self.qoi = [None] * len(self.runmodel_object.qoi_list)
        if type(self.runmodel_object.qoi_list[0]) is dict:
            for j in range(len(self.runmodel_object.qoi_list)):
                self.qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
        else:
            self.qoi = self.runmodel_object.qoi_list
