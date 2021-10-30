from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *
from UQpy.sampling.StratifiedSampling import *
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger
from UQpy.utilities.Utilities import process_random_state


class RefinedStratifiedSampling:
    """


    **Inputs:**

    * **sample_object** (``SampleMethods`` object(s)):


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

        Default: 1.

    * **nsamples** (`int`):
        .

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):


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

    @beartype
    def __init__(
        self,
        stratified_sampling: StratifiedSampling,
        refinement_algorithm: Refinement,
        samples_number: PositiveInteger = None,
        samples_per_iteration: int = 1,
        random_state: RandomStateType = None,
    ):
        """

        :param stratified_sampling: Generally, this must be an object of a ``UQpy.SampleMethods`` class. Each child
         class of ``RefinedStratifiedSampling`` has it's own constraints on which specific types of ``SampleMethods``
         it can accept. These are described in the child class documentation below.
        :param refinement_algorithm: TODO
        :param samples_number: Total number of samples to be drawn (including the initial samples).
         If `samples_number` is provided when instantiating the class, the ``run`` method will automatically be called.
         If `samples_number` is not provided, an ``RSS`` subclass can be executed by invoking the ``run`` method and
         passing `samples_number`
        :param samples_per_iteration: Number of samples to be added per iteration.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
         object itself can be passed directly.
        """
        self.stratified_sampling = stratified_sampling
        self.samples_per_iteration = samples_per_iteration
        self.refinement_algorithm = refinement_algorithm
        self.training_points = self.stratified_sampling.samplesU01
        self.samplesU01 = self.stratified_sampling.samplesU01
        self.samples = self.stratified_sampling.samples
        self.dimension = self.samples.shape[1]
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        self.samples_number = samples_number

        self.random_state = process_random_state(random_state)

        if self.samples_number is not None:
            self.run(samples_number=self.samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):
        self.samples_number = samples_number

        if self.samples_number <= self.samples.shape[0]:
            raise ValueError(
                "UQpy Error: The number of requested samples must be larger than the existing "
                "sample set."
            )

        initial_number = self.samples.shape[0]

        self.refinement_algorithm.initialize(self.samples_number, self.training_points)

        for i in range(initial_number, samples_number, self.samples_per_iteration):
            new_points = self.refinement_algorithm.update_samples(
                self.samples_number,
                self.samples_per_iteration,
                self.random_state,
                i,
                self.dimension,
                self.samplesU01,
                self.training_points,
            )
            self.append_samples(new_points)

            self.refinement_algorithm.finalize(self.samples, self.samples_per_iteration)

    def append_samples(self, new_points):
        # Adding new sample to training points, samplesU01 and samples attributes
        self.training_points = np.vstack([self.training_points, new_points])
        self.samplesU01 = np.vstack([self.samplesU01, new_points])
        new_point_ = np.zeros_like(new_points)
        for k in range(self.dimension):
            new_point_[:, k] = self.stratified_sampling.distributions[k].icdf(
                new_points[:, k]
            )
        self.samples = np.vstack([self.samples, new_point_])
