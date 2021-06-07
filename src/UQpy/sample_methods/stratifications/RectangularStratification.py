from UQpy.sample_methods.stratifications.StratifiedSampling import StratifiedSampling
from UQpy.sample_methods.strata import RectangularStrata
import numpy as np
import scipy.stats as stats

class RectangularSTS(StratifiedSampling):
    """
    Executes Stratified Sampling using Rectangular Stratification.

    ``RectangularSTS`` is a child class of ``stratifications``. ``RectangularSTS`` takes in all parameters defined in the parent
    ``stratifications`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``stratifications`` for additional details.

    **Inputs:**

    * **strata_object** (``RectangularStrata`` object):
        The `strata_object` for ``RectangularSTS`` must be an object of type ``RectangularStrata`` class.

    * **sts_criterion** (`str`):
        Random or Centered samples inside the rectangular strata.
        Options:
        1. 'random' - Samples are drawn randomly within the strata. \n
        2. 'centered' - Samples are drawn at the center of the strata. \n

        Default: 'random'

    **Methods:**

    """
    def __init__(self, distributions, strata_object, samples_per_stratum_number=None, samples_number=None, sts_criterion="random",
                 verbose=False, random_state=None):
        if not isinstance(strata_object, RectangularStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of RectangularStrata class")

        self.sts_criterion = sts_criterion
        if self.sts_criterion not in ['random', 'centered']:
            raise NotImplementedError("UQpy: Supported sts_criteria: 'random', 'centered'")
        if samples_number is not None:
            if self.sts_criterion == 'centered':
                if samples_number != len(strata_object.volume):
                    raise ValueError("UQpy: 'nsamples' attribute is not consistent with number of seeds for 'centered' "
                                     "sampling")
        if samples_per_stratum_number is not None:
            if self.sts_criterion == "centered":
                samples_per_stratum_number = [1] * strata_object.widths.shape[0]

        super().__init__(distributions=distributions, strata_object=strata_object,
                         samples_per_stratum_number=samples_per_stratum_number, samples_number=samples_number, random_state=random_state,
                         verbose=verbose)

    def create_unit_hypercube_samples(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in rectangular strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``stratifications`` class for additional details.
        """

        samples_in_strata, weights = [], []

        for i in range(self.strata_object.seeds.shape[0]):
            samples_temp = np.zeros([int(self.samples_per_stratum_number[i]), self.strata_object.seeds.shape[1]])
            for j in range(self.strata_object.seeds.shape[1]):
                if self.sts_criterion == "random":
                    samples_temp[:, j] = stats.uniform.rvs(loc=self.strata_object.seeds[i, j],
                                                           scale=self.strata_object.widths[i, j],
                                                           random_state=self.random_state,
                                                           size=int(self.samples_per_stratum_number[i]))
                else:
                    samples_temp[:, j] = self.strata_object.seeds[i, j] + self.strata_object.widths[i, j] / 2.
            samples_in_strata.append(samples_temp)

            if int(self.samples_per_stratum_number[i]) != 0:
                weights.extend(
                    [self.strata_object.volume[i] / self.samples_per_stratum_number[i]] * int(self.samples_per_stratum_number[i]))
            else:
                weights.extend([0] * int(self.samples_per_stratum_number[i]))

        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)