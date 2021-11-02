from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *
from UQpy.sampling.StratifiedSampling import *
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger
from UQpy.utilities.Utilities import process_random_state


class RefinedStratifiedSampling:
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

        :param stratified_sampling: Generally, this must be an object of a :py:mod:`UQpy.sampling`` class. Each child
         class of :class:`.RefinedStratifiedSampling` has it's own constraints on which specific types
         it can accept. These are described in the child class documentation below.
        :param refinement_algorithm: Algorithm used for the refinement of the strata. Two method exist Simple and
         Gradient Enhance Refinement.
        :param samples_number: Total number of samples to be drawn (including the initial samples).
         If `samples_number` is provided when instantiating the class, the :meth:`run` method will automatically be
         called. If `samples_number` is not provided, an :class:`.RefinedStratifiedSampling` subclass can be executed
         by invoking the :meth:`run` method and passing `samples_number`
        :param samples_per_iteration: Number of samples to be added per iteration.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.stratified_sampling = stratified_sampling
        self.samples_per_iteration = samples_per_iteration
        self.refinement_algorithm = refinement_algorithm
        self.training_points = self.stratified_sampling.samplesU01
        self.samplesU01 = self.stratified_sampling.samplesU01
        """The generated samples on the unit hypercube."""
        self.samples = self.stratified_sampling.samples
        """The generated stratified samples following the prescribed distribution."""
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
