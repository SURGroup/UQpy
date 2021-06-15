import numpy as np

from UQpy.sample_methods.stratifications.StratifiedSampling import StratifiedSampling
from UQpy.sample_methods.SimplexSampling import SimplexSampling
from UQpy.sample_methods.strata import DelaunayStrata


class DelaunaySTS(StratifiedSampling):
    """
    Executes Stratified Sampling using Delaunay Stratification.

    ``DelaunaySTS`` is a child class of ``stratifications``. ``DelaunaySTS`` takes in all parameters defined in the
      parent ``stratifications`` class with differences note below. Only those inputs and attributes that differ from
      the parent class are listed below. See documentation for ``stratifications`` for additional details.

    **Inputs:**

    * **strata_object** (``DelaunayStrata`` object):
        The `strata_object` for ``DelaunaySTS`` must be an object of the ``DelaunayStrata`` class.

    **Methods:**

    """
    def __init__(self, distributions, strata_object, samples_per_stratum_number=1, samples_number=None,
                 random_state=None, verbose=False):

        if not isinstance(strata_object, DelaunayStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of DelaunayStrata class")

        super().__init__(distributions=distributions, strata_object=strata_object,
                         samples_per_stratum_number=samples_per_stratum_number, samples_number=samples_number,
                         random_state=random_state, verbose=verbose)

    def create_unit_hypercube_samples(self, samples_per_stratum_number=None, samples_number=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in Delaunay strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``stratifications`` class for additional details.
        """

        samples_in_strata, weights = [], []
        count = 0
        for simplex in self.strata_object.delaunay.simplices:  # extract simplices from Delaunay triangulation
            samples_temp = SimplexSampling(nodes=self.strata_object.delaunay.points[simplex],
                                           samples_number=int(self.samples_per_stratum_number[count]),
                                           random_state=self.random_state)
            samples_in_strata.append(samples_temp.samples)
            self.extend_weights(count, weights)
            count = count + 1

        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)

    def extend_weights(self, index, weights):
        if int(self.samples_per_stratum_number[index]) != 0:
            weights.extend(
                [self.strata_object.volume[index] / self.samples_per_stratum_number[index]] * int(
                    self.samples_per_stratum_number[
                        index]))
        else:
            weights.extend([0] * int(self.samples_per_stratum_number[count]))
