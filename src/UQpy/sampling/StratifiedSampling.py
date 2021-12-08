import logging
from beartype import beartype
from numpy.random import RandomState

from UQpy.distributions import DistributionContinuous1D, JointIndependent
from UQpy.utilities.strata.baseclass.Strata import Strata
from UQpy.utilities.ValidationTypes import *
from UQpy.utilities.Utilities import process_random_state


class StratifiedSampling:
    @beartype
    def __init__(
        self,
        distributions: Union[DistributionContinuous1D, JointIndependent, list[DistributionContinuous1D]],
        strata_object: Strata,
        samples_per_stratum_number: Union[int, list[int]] = None,
        samples_number: int = None,
        random_state: RandomStateType = None,
    ):
        """
        Class for Stratified Sampling (:cite:`StratifiedSampling1`).

        :param distributions: List of :class:`.Distribution` objects corresponding to each random variable.
        :param strata_object: Defines the stratification of the unit hypercube. This must be provided and must be an
         object of a :class:`.Strata` child class: :class:`.Rectangular`, :class:`.Voronoi`, or :class:`.Delaunay`.
        :param samples_per_stratum_number: Specifies the number of samples in each stratum. This must be either an
         integer, in which case an equal number of samples are drawn from each stratum, or a list. If it is provided as
         a list, the length of the list must be equal to the number of strata.
         If `samples_per_stratum_number` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `samples_per_stratum_number` or `samples_number` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param samples_number: Specify the total number of samples. If `samples_number` is specified, the samples will
         be drawn in proportion to the volume of the strata. Thus, each stratum will contain
         :math:`round(V_i*samples number)` samples.
         If `samples_number` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `samples_per_stratum_number` or `samples_number` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.logger = logging.getLogger(__name__)
        self.weights = None
        """Individual sample weights."""
        self.strata_object = strata_object

        self.samples_per_stratum_number = samples_per_stratum_number
        self.samples_number = samples_number

        self.samples = None
        """The generated samples following the prescribed distribution."""
        self.samplesU01 = None
        """The generated samples on the unit hypercube."""

        self.distributions = distributions
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        if self.random_state is None:
            self.random_state = self.strata_object.random_state

        self.strata_object.check_centered(samples_number)
        self.logger.info("UQpy: Stratified_sampling object is created")

        if self.samples_per_stratum_number is not None or self.samples_number is not None:
            self.run(samples_per_stratum_number=self.samples_per_stratum_number,
                     samples_number=self.samples_number)

    def transform_samples(self, samples01):
        """
        Transform samples in the unit hypercube :math:`[0, 1]^n` to the prescribed distribution using the inverse CDF.

        :param samples01: `ndarray` containing the generated samples on [0, 1]^dimension.

        :return `ndarray` containing the generated samples following the prescribed distribution.
        """
        samples_u_to_x = np.zeros_like(samples01)
        for j in range(samples01.shape[1]):
            samples_u_to_x[:, j] = self.distributions[j].icdf(samples01[:, j])

        self.samples = samples_u_to_x

    @beartype
    def run(
        self,
        samples_per_stratum_number: Union[None, int, list[int]] = None,
        samples_number: Union[None, PositiveInteger] = None,
    ):
        """
        Executes stratified sampling.

        This method performs the sampling for each of the child classes by running two methods:
        :meth:`create_samplesu01`, and :meth:`transform_samples`. The :meth:`create_samplesu01` method is
        unique to each child class and therefore must be overwritten when a new child class is defined. The
        :meth:`transform_samples` method is common to all stratified sampling classes and is therefore defined by the
        parent class. It does not need to be modified.

        If `samples_number` or `samples_per_stratum_number` is provided when the class is defined, the :meth:`run`
        method will be executed automatically.  If neither `samples_per_stratum_number` or `samples_number` are provided
        when the class is defined, the user must call the :meth:`run` method to perform stratified sampling.

        :param samples_per_stratum_number: Specifies the number of samples in each stratum. This must be either an
         integer, in which case an equal number of samples are drawn from each stratum, or a list. If it is provided as
         a list, the length of the list must be equal to the number of strata.
         If `samples_per_stratum_number` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `samples_per_stratum_number` or `samples_number` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param samples_number: Specify the total number of samples. If `samples_number` is specified, the samples will
         be drawn in proportion to the volume of the strata. Thus, each stratum will contain
         `round(V_i*samples_number)` samples where :math:`V_i \le 1` is the volume of stratum `i` in the unit
         hypercube.
         If `samples_number` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `samples_per_stratum_number` or `samples_number` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        """

        self.samples_per_stratum_number = samples_per_stratum_number
        self.samples_number = samples_number
        self._run_checks()

        self.logger.info("UQpy: Performing Stratified Sampling")

        self.create_unit_hypercube_samples()

        self.transform_samples(self.samplesU01)

        self.logger.info("UQpy: Stratified Sampling is completed")

    def _run_checks(self):
        if self.samples_number is not None:
            self.samples_per_stratum_number = (self.strata_object.volume * self.samples_number).round()

        if self.samples_per_stratum_number is not None:
            if isinstance(self.samples_per_stratum_number, int):
                self.samples_per_stratum_number = [self.samples_per_stratum_number] * \
                                                  self.strata_object.volume.shape[0]
            elif isinstance(self.samples_per_stratum_number, list):
                if len(self.samples_per_stratum_number) != self.strata_object.volume.shape[0]:
                    raise ValueError("UQpy: Length of 'samples_per_stratum_number' must match the number of strata.")
            elif self.samples_number is None:
                raise ValueError("UQpy: 'samples_per_stratum_number' must be an integer or a list.")
        else:
            self.samples_per_stratum_number = [1] * self.strata_object.volume.shape[0]

    def create_unit_hypercube_samples(self):
        samples_in_strata, weights = self.strata_object.sample_strata(
            self.samples_per_stratum_number, self.random_state)
        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
