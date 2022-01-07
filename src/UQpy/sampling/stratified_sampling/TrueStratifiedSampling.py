import logging
from beartype import beartype
from numpy.random import RandomState

from UQpy.sampling.stratified_sampling.baseclass.StratifiedSampling import StratifiedSampling
from UQpy.distributions import DistributionContinuous1D, JointIndependent
from UQpy.sampling.stratified_sampling.strata import RectangularStrata
from UQpy.sampling.stratified_sampling.strata.baseclass.Strata import Strata
from UQpy.utilities.ValidationTypes import *


class TrueStratifiedSampling(StratifiedSampling):
    @beartype
    def __init__(
        self,
        distributions: Union[DistributionContinuous1D, JointIndependent, list[DistributionContinuous1D]],
        strata_object: Strata,
        nsamples_per_stratum: Union[int, list[int]] = None,
        nsamples: int = None,
        random_state: RandomStateType = None,
    ):
        """
        Class for Stratified Sampling (:cite:`StratifiedSampling1`).

        :param distributions: List of :class:`.Distribution` objects corresponding to each random variable.
        :param strata_object: Defines the stratification of the unit hypercube. This must be provided and must be an
         object of a :class:`.Strata` child class: :class:`.Rectangular`, :class:`.Voronoi`, or :class:`.Delaunay`.
        :param nsamples_per_stratum: Specifies the number of samples in each stratum. This must be either an
         integer, in which case an equal number of samples are drawn from each stratum, or a list. If it is provided as
         a list, the length of the list must be equal to the number of strata.
         If `nsamples_per_stratum` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param nsamples: Specify the total number of samples. If `nsamples` is specified, the samples will
         be drawn in proportion to the volume of the strata. Thus, each stratum will contain
         :code:`round(V_i* nsamples)` samples.
         If `nsamples` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.logger = logging.getLogger(__name__)
        self.weights: NumpyFloatArray = None
        """Individual sample weights."""
        self.strata_object = strata_object

        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples

        self.samples:NumpyFloatArray = None
        """The generated samples following the prescribed distribution."""
        self.samplesU01:NumpyFloatArray = None
        """The generated samples on the unit hypercube."""

        self.distributions = distributions
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        if self.random_state is None:
            self.random_state = self.strata_object.random_state

        if isinstance(self.strata_object, RectangularStrata):
            self.strata_object.check_centered(nsamples)
        self.logger.info("UQpy: Stratified_sampling object is created")

        if self.nsamples_per_stratum is not None or self.nsamples is not None:
            self.run(nsamples_per_stratum=self.nsamples_per_stratum,
                     nsamples=self.nsamples)

    def transform_samples(self, samples01):
        """
        Transform samples in the unit hypercube :math:`[0, 1]^n` to the prescribed distribution using the inverse CDF.

        :param samples01: :class:`numpy.ndarray` containing the generated samples on :math:`[0, 1]^n`.

        :return: :class:`numpy.ndarray` containing the generated samples following the prescribed distribution.
        """
        samples_u_to_x = np.zeros_like(samples01)
        for j in range(samples01.shape[1]):
            samples_u_to_x[:, j] = self.distributions[j].icdf(samples01[:, j])

        self.samples = samples_u_to_x

    @beartype
    def run(
        self,
        nsamples_per_stratum: Union[None, int, list[int]] = None,
        nsamples: Union[None, PositiveInteger] = None,
    ):
        """
        Executes stratified sampling.

        This method performs the sampling for each of the child classes by running two methods:
        :meth:`create_samplesu01`, and :meth:`transform_samples`. The :meth:`create_samplesu01` method is
        unique to each child class and therefore must be overwritten when a new child class is defined. The
        :meth:`transform_samples` method is common to all stratified sampling classes and is therefore defined by the
        parent class. It does not need to be modified.

        If `nsamples` or `nsamples_per_stratum` is provided when the class is defined, the :meth:`run`
        method will be executed automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided
        when the class is defined, the user must call the :meth:`run` method to perform stratified sampling.

        :param nsamples_per_stratum: Specifies the number of samples in each stratum. This must be either an
         integer, in which case an equal number of samples are drawn from each stratum, or a list. If it is provided as
         a list, the length of the list must be equal to the number of strata.
         If `nsamples_per_stratum` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        :param nsamples: Specify the total number of samples. If `nsamples` is specified, the samples will
         be drawn in proportion to the volume of the strata. Thus, each stratum will contain
         :code:`round(V_i*nsamples)` samples where :math:`V_i \le 1` is the volume of stratum `i` in the unit
         hypercube.
         If `nsamples` is provided when the class is defined, the :meth:`run` method will be executed
         automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is
         defined, the user must call the :meth:`run` method to perform stratified sampling.
        """

        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples
        self._run_checks()

        self.logger.info("UQpy: Performing Stratified Sampling")

        self.create_unit_hypercube_samples()

        self.transform_samples(self.samplesU01)

        self.logger.info("UQpy: Stratified Sampling is completed")

    def _run_checks(self):
        if self.nsamples is not None:
            self.nsamples_per_stratum = (self.strata_object.volume * self.nsamples).round()

        if self.nsamples_per_stratum is not None:
            if isinstance(self.nsamples_per_stratum, int):
                self.nsamples_per_stratum = [self.nsamples_per_stratum] * \
                                                  self.strata_object.volume.shape[0]
            elif isinstance(self.nsamples_per_stratum, list):
                if len(self.nsamples_per_stratum) != self.strata_object.volume.shape[0]:
                    raise ValueError("UQpy: Length of 'nsamples_per_stratum' must match the number of strata.")
            elif self.nsamples is None:
                raise ValueError("UQpy: 'nsamples_per_stratum' must be an integer or a list.")
        else:
            self.nsamples_per_stratum = [1] * self.strata_object.volume.shape[0]

    def create_unit_hypercube_samples(self):
        samples_in_strata, weights = self.strata_object.sample_strata(
            self.nsamples_per_stratum, self.random_state)
        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
