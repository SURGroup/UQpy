from UQpy.SampleMethods.STS.sts import STS
from UQpy.SampleMethods.Strata import RectangularStrata
import numpy as np
import scipy.stats as stats


class RectangularSTS(STS):
    """
    Executes Stratified Sampling using Rectangular Stratification.

    ``RectangularSTS`` is a child class of ``STS``. ``RectangularSTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

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
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, sts_criterion="random",
                 verbose=False, random_state=None):
        if not isinstance(strata_object, RectangularStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of RectangularStrata class")

        self.sts_criterion = sts_criterion
        if self.sts_criterion not in ['random', 'centered']:
            raise NotImplementedError("UQpy: Supported sts_criteria: 'random', 'centered'")
        if nsamples is not None:
            if self.sts_criterion == 'centered':
                if nsamples != len(strata_object.volume):
                    raise ValueError("UQpy: 'nsamples' attribute is not consistent with number of seeds for 'centered' "
                                     "sampling")
        if nsamples_per_stratum is not None:
            if self.sts_criterion == "centered":
                nsamples_per_stratum = [1] * strata_object.widths.shape[0]

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """

        Overwrites the ``create_samplesu01`` method in the parent class.

        This method generate samples in rectangular strata on the unit hypercube. It has the same inputs and outputs as
        the ``create_samplesu01`` method in the parent class. See the ``STS`` class for additional details.

        """
        samples_in_strata, weights = [], []

        for i in range(self.strata_object.seeds.shape[0]):
            samples_temp = np.zeros([int(self.nsamples_per_stratum[i]), self.strata_object.seeds.shape[1]])
            for j in range(self.strata_object.seeds.shape[1]):
                if self.sts_criterion == "random":
                    samples_temp[:, j] = stats.uniform.rvs(loc=self.strata_object.seeds[i, j],
                                                           scale=self.strata_object.widths[i, j],
                                                           random_state=self.random_state,
                                                           size=int(self.nsamples_per_stratum[i]))
                else:
                    samples_temp[:, j] = self.strata_object.seeds[i, j] + self.strata_object.widths[i, j] / 2.
            samples_in_strata.append(samples_temp)
            if int(self.nsamples_per_stratum[i]) != 0:
                weights.extend(
                    [self.strata_object.volume[i] / self.nsamples_per_stratum[i]] * int(self.nsamples_per_stratum[i]))
            else:
                weights.extend([0] * int(self.nsamples_per_stratum[i]))
        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
