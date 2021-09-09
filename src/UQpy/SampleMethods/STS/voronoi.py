import numpy as np

from UQpy.SampleMethods.STS.sts import STS
from UQpy.SampleMethods.Simplex import Simplex
from UQpy.SampleMethods.Strata import VoronoiStrata


class VoronoiSTS(STS):
    """
    Executes Stratified Sampling using Voronoi Stratification.

    ``VoronoiSTS`` is a child class of ``STS``. ``VoronoiSTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

    **Inputs:**

    * **strata_object** (``VoronoiStrata`` object):
        The `strata_object` for ``VoronoiSTS`` must be an object of the ``VoronoiStrata`` class.

    **Methods:**

    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, random_state=None,
                 verbose=False):
        # Check strata_object
        if not isinstance(strata_object, VoronoiStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of VoronoiStrata class")

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in Voronoi strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``STS`` class for additional details.
        """
        from scipy.spatial import Delaunay, ConvexHull

        samples_in_strata, weights = list(), list()
        for j in range(len(self.strata_object.vertices)):  # For each bounded region (Voronoi stratification)
            vertices = self.strata_object.vertices[j][:-1, :]
            seed = self.strata_object.seeds[j, :].reshape(1, -1)
            seed_and_vertices = np.concatenate([vertices, seed])

            # Create Dealunay Triangulation using seed and vertices of each stratum
            delaunay_obj = Delaunay(seed_and_vertices)

            # Compute volume of each delaunay
            volume = list()
            for i in range(len(delaunay_obj.vertices)):
                vert = delaunay_obj.vertices[i]
                ch = ConvexHull(seed_and_vertices[vert])
                volume.append(ch.volume)

            temp_prob = np.array(volume) / sum(volume)
            a = list(range(len(delaunay_obj.vertices)))
            for k in range(int(self.nsamples_per_stratum[j])):
                simplex = self.random_state.choice(a, p=temp_prob)

                new_samples = Simplex(nodes=seed_and_vertices[delaunay_obj.vertices[simplex]], nsamples=1,
                                      random_state=self.random_state).samples

                samples_in_strata.append(new_samples)

            if int(self.nsamples_per_stratum[j]) != 0:
                weights.extend(
                    [self.strata_object.volume[j] / self.nsamples_per_stratum[j]] * int(self.nsamples_per_stratum[j]))
            # else:
            #     weights.extend([0] * int(self.nsamples_per_stratum[j]))

        self.weights = weights
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)
