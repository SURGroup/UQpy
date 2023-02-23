
from UQpy.sampling.stratified_sampling.TrueStratifiedSampling import *
from UQpy.sampling.stratified_sampling.refinement.baseclass import Refinement
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger


class RefinedStratifiedSampling(StratifiedSampling):
    @beartype
    def __init__(
        self,
        stratified_sampling: Union[TrueStratifiedSampling],
        refinement_algorithm: Refinement,
        nsamples: PositiveInteger = None,
        samples_per_iteration: int = 1,
        random_state: RandomStateType = None,
    ):
        """

        :param stratified_sampling: Generally, this must be an object of a :class:`.TrueStratifiedSampling` class.
        :param refinement_algorithm: Algorithm used for the refinement of the strata. Two methods exist
         :class:`.RandomRefinement` and :class:`.GradientEnhancedRefinement` .
        :param nsamples: Total number of samples to be drawn (including the initial samples).
         If `nsamples` is provided when instantiating the class, the :meth:`run` method will automatically be
         called. If `nsamples` is not provided, :class:`.RefinedStratifiedSampling` can be executed
         by invoking the :meth:`run` method and passing `nsamples`
        :param samples_per_iteration: Number of samples to be added per refinement iteration.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.stratified_sampling = stratified_sampling
        self.samples_per_iteration = samples_per_iteration
        self.refinement_algorithm = refinement_algorithm
        self.refinement_algorithm.update_strata(stratified_sampling.samplesU01)
        self.training_points = self.stratified_sampling.samplesU01
        self.samplesU01: Numpy2DFloatArray = self.stratified_sampling.samplesU01
        """The generated samples on the unit hypercube."""
        self.samples: Numpy2DFloatArray = self.stratified_sampling.samples
        """The generated stratified samples following the prescribed distribution."""
        self.dimension = self.samples.shape[1]
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        self.nsamples = nsamples

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.default_rng(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.Generator object.')
        if self.random_state is None:
            self.random_state = self.stratified_sampling.random_state

        if self.nsamples is not None:
            self.run(nsamples=self.nsamples)

    @beartype
    def run(self, nsamples: PositiveInteger):
        """
        Executes refined stratified sampling.

        :param nsamples:  Total number of samples to be drawn (including the initial samples).
         If `nsamples` is provided when instantiating the class, the :meth:`run` method will automatically be
         called. If `nsamples` is not provided, an :class:`.RefinedStratifiedSampling` subclass can be executed
         by invoking the :meth:`run` method and passing `nsamples`
        """
        self.nsamples = nsamples

        if self.nsamples <= self.samples.shape[0]:
            raise ValueError("UQpy Error: The number of requested samples must be larger than the existing "
                             "sample set.")

        initial_number = self.samples.shape[0]

        self.refinement_algorithm.initialize(self.nsamples, self.training_points, self.samples)

        for i in range(initial_number, nsamples, self.samples_per_iteration):
            new_points = self.refinement_algorithm.update_samples(
                self.nsamples, self.samples_per_iteration,
                self.random_state, i, self.dimension,
                self.samplesU01, self.training_points)
            self.append_samples(new_points)

            self.refinement_algorithm.finalize(self.samples, self.samples_per_iteration)

    def append_samples(self, new_points):
        # Adding new sample to training points, samplesU01 and samples attributes
        self.training_points = np.vstack([self.training_points, new_points])
        self.samplesU01 = np.vstack([self.samplesU01, new_points])
        new_point_ = np.zeros_like(new_points)
        for k in range(self.dimension):
            new_point_[:, k] = self.stratified_sampling.distributions[k].icdf(new_points[:, k])
        self.samples = np.vstack([self.samples, new_point_])
