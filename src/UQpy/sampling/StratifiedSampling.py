import logging
from typing import List
from beartype import beartype
from UQpy.distributions import DistributionContinuous1D, JointIndependent
from UQpy.utilities.strata.baseclass import Strata
from UQpy.utilities.ValidationTypes import *


class StratifiedSampling:

    @beartype
    def __init__(self,
                 distributions: Union[DistributionContinuous1D, JointIndependent, List[DistributionContinuous1D]],
                 strata_object: Strata,
                 samples_per_stratum_number: Union[int, List[int]] = None,
                 samples_number: int = None):

        self.logger = logging.getLogger(__name__)
        self.weights = None
        self.strata_object = strata_object
        self.samples_per_stratum_number = samples_per_stratum_number
        self.samples_number = samples_number
        self.samplesU01, self.samples = None, None
        self.distributions = distributions

        self.logger.info("UQpy: Stratified_sampling object is created")

        if self.samples_per_stratum_number is not None or self.samples_number is not None:
            self.run(samples_per_stratum_number=self.samples_per_stratum_number, samples_number=self.samples_number)

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
            samples_u_to_x[:, j] = self.distributions[j].icdf(samples01[:, j])

        self.samples = samples_u_to_x

    @beartype
    def run(self, samples_per_stratum_number: Union[None, int, List[int]] = None,
            samples_number: Union[None, PositiveInteger] = None):
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
                self.samples_per_stratum_number = [self.samples_per_stratum_number] * self.strata_object.volume.shape[0]
            elif isinstance(self.samples_per_stratum_number, list):
                if len(self.samples_per_stratum_number) != self.strata_object.volume.shape[0]:
                    raise ValueError("UQpy: Length of 'nsamples_per_stratum' must match the number of strata.")
            elif self.samples_number is None:
                raise ValueError("UQpy: 'nsamples_per_stratum' must be an integer or a list.")
        else:
            self.samples_per_stratum_number = [1] * self.strata_object.volume.shape[0]

    def create_unit_hypercube_samples(self):
        samples_in_strata, weights = self.strata_object.sample_strata(self.samples_per_stratum_number)
        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
