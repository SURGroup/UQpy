import numpy as np


class STS:
    """
    Parent class for Stratified Sampling ([9]_).

    This is the parent class for all stratified sampling methods. This parent class only provides the framework for
    stratified sampling and cannot be used directly for the sampling. Sampling is done by calling the child
    class for the desired stratification.

    **Inputs:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **strata_object** (``Strata`` object)
        Defines the stratification of the unit hypercube. This must be provided and must be an object of a ``Strata``
        child class: ``RectangularStrata``, ``VoronoiStrata``, or ``DelaunayStrata``.

    * **nsamples_per_stratum** (`int` or `list`):
        Specifies the number of samples in each stratum. This must be either an integer, in which case an equal number
        of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the list must be
        equal to the number of strata.

        If `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be executed
        automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the user
        must call the ``run`` method to perform stratified sampling.

    * **nsamples** (`int`):
        Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
        the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples.

        If `nsamples` is provided when the class is defined, the ``run`` method will be executed
        automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the user
        must call the ``run`` method to perform stratified sampling.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False

    **Attributes:**

    * **samples** (`ndarray`):
        The generated samples following the prescribed distribution.

    * **samplesU01** (`ndarray`)
        The generated samples on the unit hypercube.

    * **weights** (`ndarray`)
        Individual sample weights.

    **Methods:**
    """

    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, random_state=None,
                 verbose=False):

        self.verbose = verbose
        self.weights = None
        self.strata_object = strata_object
        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples
        self.samplesU01, self.samples = None, None

        # Check if a Distribution object is provided.
        from UQpy.Distributions import DistributionContinuous1D, JointInd

        if isinstance(dist_object, list):
            self.dimension = len(dist_object)
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            self.dimension = 1
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.dist_object = dist_object

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        if self.random_state is None:
            self.random_state = self.strata_object.random_state

        if self.verbose:
            print("UQpy: STS object is created")

        # If nsamples_per_stratum or nsamples is provided, execute run method
        if self.nsamples_per_stratum is not None or self.nsamples is not None:
            self.run(nsamples_per_stratum=self.nsamples_per_stratum, nsamples=self.nsamples)

    def transform_samples(self, samples01):
        """
        Transform samples in the unit hypercube :math:`[0, 1]^n` to the prescribed distribution using the inverse CDF.

        **Inputs:**

        * **samplesU01** (`ndarray`):
            `ndarray` containing the generated samples on [0, 1]^dimension.

        **Outputs:**

        * **samples** (`ndarray`):
            `ndarray` containing the generated samples following the prescribed distribution.
        """

        samples_u_to_x = np.zeros_like(samples01)
        for j in range(0, samples01.shape[1]):
            samples_u_to_x[:, j] = self.dist_object[j].icdf(samples01[:, j])

        self.samples = samples_u_to_x

    def run(self, nsamples_per_stratum=None, nsamples=None):
        """
        Executes stratified sampling.

        This method performs the sampling for each of the child classes by running two methods:
        ``create_samplesu01``, and ``transform_samples``. The ``create_samplesu01`` method is
        unique to each child class and therefore must be overwritten when a new child class is defined. The
        ``transform_samples`` method is common to all stratified sampling classes and is therefore defined by the parent
        class. It does not need to be modified.

        If `nsamples` or `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be
        executed automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined,
        the user must call the ``run`` method to perform stratified sampling.

        **Input:**

        * **nsamples_per_stratum** (`int` or `list`):
            Specifies the number of samples in each stratum. This must be either an integer, in which case an equal
            number of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the
            list must be equal to the number of strata.

            If `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be executed
            automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the
            user must call the ``run`` method to perform stratified sampling.

        * **nsamples** (`int`):
            Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
            the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples where
            :math:`V_i \le 1` is the volume of stratum `i` in the unit hypercube.

            If `nsamples` is provided when the class is defined, the ``run`` method will be executed
            automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the
            user must call the ``run`` method to perform stratified sampling.

        **Outputs:**

        The ``run`` method has no output, although it modifies the `samples`, `samplesu01`, and `weights` attributes.
        """

        # Check inputs of run methods
        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples
        self._run_checks()

        if self.verbose:
            print("UQpy: Performing Stratified Sampling")

        # Call "create_sampleu01" method and generate samples in  the unit hypercube
        self.create_samplesu01(nsamples_per_stratum, nsamples)

        # Compute inverse cdf of samplesU01
        self.transform_samples(self.samplesU01)

        if self.verbose:
            print("UQpy: Stratified Sampling is completed")

    def _run_checks(self):
        if self.nsamples is not None:
            if not isinstance(self.nsamples, int):
                raise RuntimeError("UQpy: 'nsamples' must be an integer.")
            else:
                if isinstance(self.nsamples_per_stratum, (int, list)):
                    print("UQpy: STS class is executing proportional sampling, thus ignoring "
                          "'nsamples_per_stratum'.")
            self.nsamples_per_stratum = (self.strata_object.volume * self.nsamples).round()

        if self.nsamples_per_stratum is not None:
            if isinstance(self.nsamples_per_stratum, int):
                self.nsamples_per_stratum = [self.nsamples_per_stratum] * self.strata_object.volume.shape[0]
            elif isinstance(self.nsamples_per_stratum, list):
                if len(self.nsamples_per_stratum) != self.strata_object.volume.shape[0]:
                    raise ValueError("UQpy: Length of 'nsamples_per_stratum' must match the number of strata.")
            elif self.nsamples is None:
                raise ValueError("UQpy: 'nsamples_per_stratum' must be an integer or a list.")
        else:
            self.nsamples_per_stratum = [1] * self.strata_object.volume.shape[0]

    # Creating dummy method for create_samplesU01. These methods are overwritten in child classes.
    def create_samplesu01(self, nsamples_per_stratum, nsamples):
        """
        Executes the specific stratified sampling algorithm. This method is overwritten by each child class of ``STS``.

        **Input:**

        * **nsamples_per_stratum** (`int` or `list`):
            Specifies the number of samples in each stratum. This must be either an integer, in which case an equal
            number of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the
            list must be equal to the number of strata.

            Either `nsamples_per_stratum` or `nsamples` must be provided.

        * **nsamples** (`int`):
            Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
            the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples where
            :math:`V_i \le 1` is the volume of stratum `i` in the unit hypercube.

            Either `nsamples_per_stratum` or `nsamples` must be provided.

        **Outputs:**

        The ``create_samplesu01`` method has no output, although it modifies the `samplesu01` and `weights` attributes.
        """
        return None
