import numpy as np

from UQpy.SampleMethods.STS.sts import STS
from UQpy.SampleMethods.Simplex import Simplex
from UQpy.SampleMethods.Strata import DelaunayStrata


class DelaunaySTS(STS):
    """
    Executes Stratified Sampling using Delaunay Stratification.

    ``DelaunaySTS`` is a child class of ``STS``. ``DelaunaySTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

    **Inputs:**

    * **strata_object** (``DelaunayStrata`` object):
        The `strata_object` for ``DelaunaySTS`` must be an object of the ``DelaunayStrata`` class.

    **Methods:**

    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=1, nsamples=None, random_state=None,
                 verbose=False):

        if not isinstance(strata_object, DelaunayStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of DelaunayStrata class")

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in Delaunay strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``STS`` class for additional details.
        """

        samples_in_strata, weights = [], []
        count = 0
        for simplex in self.strata_object.delaunay.simplices:  # extract simplices from Delaunay triangulation
            if int(self.nsamples_per_stratum[count]) != 0:
                samples_temp = Simplex(nodes=self.strata_object.delaunay.points[simplex],
                                       nsamples=int(self.nsamples_per_stratum[count]), random_state=self.random_state)
                samples_in_strata.append(samples_temp.samples)
                weights.extend(
                    [self.strata_object.volume[count] / self.nsamples_per_stratum[count]] * int(
                        self.nsamples_per_stratum[
                            count]))
            # else:
            #     weights.extend([0] * int(self.nsamples_per_stratum[count]))
            count = count + 1

        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
