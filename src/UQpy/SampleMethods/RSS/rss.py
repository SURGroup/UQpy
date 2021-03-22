import numpy as np
from UQpy.Utilities import gradient

class RSS:
    """
    Parent class for Refined Stratified Sampling [10]_, [11]_.

    This is the parent class for all refined stratified sampling methods. This parent class only provides the
    framework for refined stratified sampling and cannot be used directly for the sampling. Sampling is done by
    calling the child class for the desired algorithm.

    **Inputs:**

    * **sample_object** (``SampleMethods`` object(s)):
        Generally, this must be an object of a ``UQpy.SampleMethods`` class. Each child class of ``RSS`` has it's
        own constraints on which specific types of ``SampleMethods`` it can accept. These are described in the child
        class documentation below.

    * **runmodel_object** (``RunModel`` object):
        A ``RunModel`` object, which is used to evaluate the model.

        `runmodel_object` is optional. If it is provided, the specific ``RSS`` subclass with use it to compute the
        gradient of the model in each stratum for gradient-enhanced refined stratified sampling. If it is not
        provided, the ``RSS`` subclass will default to random stratum refinement.

    * **krig_object** (`class` object):
        A object defining a Kriging surrogate model, this object must have ``fit`` and ``predict`` methods.

        May be an object of the ``UQpy`` ``Kriging`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

        `krig_object` is only used to compute the gradient in gradient-enhanced refined stratified sampling. It must
        be provided if a `runmodel_object` is provided.

    * **local** (`Boolean`):
        In gradient enhanced refined stratified sampling, the gradient is updated after each new sample is added.
        This parameter is used to determine whether the gradient is updated for every stratum or only locally in the
        strata nearest the refined stratum.

        If `local = True`, gradients are only updated in localized regions around the refined stratum.

        Used only in gradient-enhanced refined stratified sampling.

    * **max_train_size** (`int`):
        In gradient enhanced refined stratified sampling, if `local=True` `max_train_size` specifies the number of
        nearest points at which to update the gradient.

        Used only in gradient-enhanced refined stratified sampling.

    * **step_size** (`float`)
        Defines the size of the step to use for gradient estimation using central difference method.

        Used only in gradient-enhanced refined stratified sampling.

    * **qoi_name** (`dict`):
        Name of the quantity of interest from the `runmodel_object`. If the quantity of interest is a dictionary,
        this is used to convert it to a list

        Used only in gradient-enhanced refined stratified sampling.

    * **n_add** (`int`):
        Number of samples to be added per iteration.

        Default: 1.

    * **nsamples** (`int`):
        Total number of samples to be drawn (including the initial samples).

        If `nsamples` is provided when instantiating the class, the ``run`` method will automatically be called. If
        `nsamples` is not provided, an ``RSS`` subclass can be executed by invoking the ``run`` method and passing
        `nsamples`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False

    **Attributes:**

    Each of the above inputs are saved as attributes, in addition to the following created attributes.

        * **samples** (`ndarray`):
            The generated stratified samples following the prescribed distribution.

        * **samplesU01** (`ndarray`)
            The generated samples on the unit hypercube.

        * **weights** (`ndarray`)
            Individual sample weights.

        * **strata_object** (Object of ``Strata`` subclass)
            Defines the stratification of the unit hypercube. This is an object of the ``Strata`` subclass
            corresponding to the appropriate strata type.

        **Methods:**
        """
    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        # Initialize attributes that are common to all approaches
        self.sample_object = sample_object
        self.runmodel_object = runmodel_object
        self.verbose = verbose
        self.nsamples = nsamples
        self.training_points = self.sample_object.samplesU01
        self.samplesU01 = self.sample_object.samplesU01
        self.samples = self.sample_object.samples
        self.weights = None
        self.dimension = self.samples.shape[1]
        self.n_add = n_add

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if self.runmodel_object is not None:
            if type(self.runmodel_object).__name__ not in ['RunModel']:
                raise NotImplementedError("UQpy Error: runmodel_object must be an object of the RunModel class.")

        if runmodel_object is not None:
            self.local = local
            self.max_train_size = max_train_size
            if krig_object is not None:
                if hasattr(krig_object, 'fit') and hasattr(krig_object, 'predict'):
                    self.krig_object = krig_object
                else:
                    raise NotImplementedError("UQpy Error: krig_object must have 'fit' and 'predict' methods.")
            self.qoi_name = qoi_name
            self.step_size = step_size
            if self.verbose:
                print('UQpy: GE-RSS - Running the initial sample set.')
            self.runmodel_object.run(samples=self.samples)
            if self.verbose:
                print('UQpy: GE-RSS - A RSS class object has been initiated.')
        else:
            if self.verbose:
                print('UQpy: RSS - A RSS class object has been initiated.')

        if self.nsamples is not None:
            if isinstance(self.nsamples, int) and self.nsamples > 0:
                self.run(nsamples=self.nsamples)
            else:
                raise NotImplementedError("UQpy: nsamples msut be a positive integer.")

    def run(self, nsamples):
        """
        Execute the random sampling in the ``RSS`` class.

        The ``run`` method is the function that performs random sampling in any ``RSS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``RSS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``RSS`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        The ``run`` method is inherited from the parent class and should not be modified by the subclass. It operates by
        calling a ``run_rss`` method that is uniquely defined for each subclass. All ``RSS`` subclasses must posses a
        ``run_rss`` method as defined below.

        **Input:**

        * **nsamples** (`int`):
            Total number of samples to be drawn.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Return:**

        The ``run`` method has no returns, although it creates and/or appends the `samples`, `samplesU01, `weights`, and
        `strata_object` attributes of the ``RSS`` class.
        """
        if isinstance(nsamples, int) and nsamples > 0:
            self.nsamples = nsamples
        else:
            raise RuntimeError("UQpy: nsamples must be a positive integer.")

        if self.nsamples <= self.samples.shape[0]:
            raise NotImplementedError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')

        self.run_rss()

    def estimate_gradient(self, x, y, xt):
        """
        Estimating gradients with a Kriging metamodel (surrogate).

        **Inputs:**

        * **x** (`ndarray`):
            Samples in the training data.

        * **y** (`ndarray`):
            Function values evaluated at the samples in the training data.

        * **xt** (`ndarray`):
            Samples where gradients need to be evaluated.

        **Outputs:**

        * **gr** (`ndarray`):
            First-order gradient evaluated at the points 'xt' using central difference.
        """
        if self.krig_object is not None:
            self.krig_object.fit(x, y)
            self.krig_object.nopt = 1
            tck = self.krig_object.predict
        else:
            from scipy.interpolate import LinearNDInterpolator
            tck = LinearNDInterpolator(x, y, fill_value=0).__call__

        gr = gradient(point=xt, runmodel_object=tck, order='first', df_step=self.step_size)
        return gr

    def update_samples(self, new_point):
        # Adding new sample to training points, samplesU01 and samples attributes
        self.training_points = np.vstack([self.training_points, new_point])
        self.samplesU01 = np.vstack([self.samplesU01, new_point])
        new_point_ = np.zeros_like(new_point)
        for k in range(self.dimension):
            new_point_[:, k] = self.sample_object.dist_object[k].icdf(new_point[:, k])
        self.samples = np.vstack([self.samples, new_point_])

    def identify_bins(self, strata_metric, p_):
        bin2break_, p_left = np.array([]), p_
        while np.where(strata_metric == strata_metric.max())[0].shape[0] < p_left:
            t = np.where(strata_metric == strata_metric.max())[0]
            bin2break_ = np.hstack([bin2break_, t])
            strata_metric[t] = 0
            p_left -= t.shape[0]

        tmp = self.random_state.choice(np.where(strata_metric == strata_metric.max())[0], p_left, replace=False)
        bin2break_ = np.hstack([bin2break_, tmp])
        bin2break_ = list(map(int, bin2break_))
        return bin2break_

    def run_rss(self):
        """
        This method is overwritten by each subclass in order to perform the refined stratified sampling.

        This must be an instance method of the class and, although it has no returns it should appropriately modify the
        following attributes of the class: `samples`, `samplesU01`, `weights`, `strata_object`.
        """

        pass
