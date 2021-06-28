import numpy as np
import scipy.stats as stats

from UQpy.utilities.strata.baseclass.Strata import Strata
from UQpy.utilities.strata.StratificationCriterion import StratificationCriterion
from UQpy.sampling.SimplexSampling import *


class Voronoi(Strata):
    def __init__(self, seeds=None, seeds_number=None, dimension=None, decomposition_iterations=1,
                 random_state=None, verbose=False, stratification_criterion=StratificationCriterion.RANDOM):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.seeds_number = seeds_number
        self.dimension = dimension
        self.decomposition_iterations = decomposition_iterations
        self.voronoi = None
        self.vertices = []
        self.stratification_criterion = stratification_criterion

        if self.seeds is not None:
            if self.seeds_number is not None or self.dimension is not None:
                print("UQpy: Ignoring 'nseeds' and 'dimension' attributes because 'seeds' are provided")
            self.dimension = self.seeds.shape[1]

        self.stratify()

    def stratify(self):
        """
        Performs the Voronoi stratification.
        """
        if self.verbose:
            print('UQpy: Creating Voronoi stratification ...')

        initial_seeds = self.seeds
        if self.seeds is None:
            initial_seeds = stats.uniform.rvs(size=[self.seeds_number, self.dimension], random_state=self.random_state)

        if self.decomposition_iterations == 0:
            cent, vol = self.create_volume(initial_seeds)
            self.volume = np.asarray(vol)
        else:
            for i in range(self.decomposition_iterations):
                cent, vol = self.create_volume(initial_seeds)
                initial_seeds = np.asarray(cent)
                self.volume = np.asarray(vol)

        self.seeds = initial_seeds

        if self.verbose:
            print('UQpy: Voronoi stratification created.')

    def create_volume(self, initial_seeds):
        self.voronoi, bounded_regions = self.voronoi_unit_hypercube(initial_seeds)
        cent, vol = [], []
        for region in bounded_regions:
            vertices = self.voronoi.vertices[region + [region[0]], :]
            centroid, volume = self.compute_voronoi_centroid_volume(vertices)
            self.vertices.append(vertices)
            cent.append(centroid[0, :])
            vol.append(volume)
        return cent, vol

    @staticmethod
    def voronoi_unit_hypercube(seeds):
        """
        This function reflects the seeds across all faces of the unit hypercube and creates a Voronoi decomposition of
        using all the points and their reflections. This allows a Voronoi decomposition that is bounded on the unit
        hypercube to be extracted.

        **Inputs:**

        * **seeds** (`ndarray`):
            Coordinates of points in the unit hypercube from which to define the Voronoi decomposition.

        **Output/Returns:**

        * **vor** (``scipy.spatial.Voronoi`` object):
            Voronoi decomposition of the complete set of points and their reflections.

        * **bounded_regions** (see `regions` attribute of ``scipy.spatial.Voronoi``)
            Indices of the Voronoi vertices forming each Voronoi region for those regions lying inside the unit
            hypercube.
        """

        from scipy.spatial import Voronoi

        # Mirror the seeds in both low and high directions for each dimension
        bounded_points = seeds
        dimension = seeds.shape[1]
        for i in range(dimension):
            seeds_del = np.delete(bounded_points, i, 1)
            if i == 0:
                points_temp1 = np.hstack([np.atleast_2d(-bounded_points[:, i]).T, seeds_del])
                points_temp2 = np.hstack([np.atleast_2d(2 - bounded_points[:, i]).T, seeds_del])
            elif i == dimension - 1:
                points_temp1 = np.hstack([seeds_del, np.atleast_2d(-bounded_points[:, i]).T])
                points_temp2 = np.hstack([seeds_del, np.atleast_2d(2 - bounded_points[:, i]).T])
            else:
                points_temp1 = np.hstack(
                    [seeds_del[:, :i], np.atleast_2d(-bounded_points[:, i]).T, seeds_del[:, i:]])
                points_temp2 = np.hstack(
                    [seeds_del[:, :i], np.atleast_2d(2 - bounded_points[:, i]).T, seeds_del[:, i:]])
            seeds = np.append(seeds, points_temp1, axis=0)
            seeds = np.append(seeds, points_temp2, axis=0)

        vor = Voronoi(seeds, incremental=True)

        regions = [None] * bounded_points.shape[0]

        for i in range(bounded_points.shape[0]):
            regions[i] = vor.regions[vor.point_region[i]]

        bounded_regions = regions

        return vor, bounded_regions

    @staticmethod
    def compute_voronoi_centroid_volume(vertices):
        """
        This function computes the centroid and volume of a Voronoi cell from its vertices.

        **Inputs:**

        * **vertices** (`ndarray`):
            Coordinates of the vertices that define the Voronoi cell.

        **Output/Returns:**

        * **centroid** (`ndarray`):
            Centroid of the Voronoi cell.

        * **volume** (`ndarray`):
            Volume of the Voronoi cell.
        """

        from scipy.spatial import Delaunay, ConvexHull

        tess = Delaunay(vertices)
        dimension = np.shape(vertices)[1]

        w = np.zeros((tess.nsimplex, 1))
        cent = np.zeros((tess.nsimplex, dimension))
        for i in range(tess.nsimplex):
            # pylint: disable=E1136
            ch = ConvexHull(tess.points[tess.simplices[i]])
            w[i] = ch.volume
            cent[i, :] = np.mean(tess.points[tess.simplices[i]], axis=0)

        volume = np.sum(w)
        centroid = np.matmul(np.divide(w, volume).T, cent)

        return centroid, volume

    def sample_strata(self, samples_per_stratum_number):
        from scipy.spatial import Delaunay, ConvexHull

        samples_in_strata, weights = list(), list()
        for j in range(len(self.vertices)):  # For each bounded region (Voronoi stratification)
            vertices = self.vertices[j][:-1, :]
            seed = self.seeds[j, :].reshape(1, -1)
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
            for k in range(int(samples_per_stratum_number[j])):
                simplex = self.random_state.choice(a, p=temp_prob)

                new_samples = SimplexSampling(nodes=seed_and_vertices[delaunay_obj.vertices[simplex]], samples_number=1,
                                              random_state=self.random_state).samples
                samples_in_strata.append(new_samples)

            self.extend_weights(samples_per_stratum_number, j, weights)
        return samples_in_strata, weights

    def compute_centroids(self):
        if self.mesh is None:
            self.add_boundary_points_and_construct_delaunay()
        self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
        self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
        from scipy.spatial import qhull, ConvexHull
        for j in range(self.mesh.nsimplex):
            try:
                ConvexHull(self.points[self.mesh.vertices[j]])
                self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                    Delaunay.compute_delaunay_centroid_volume(self.points[self.mesh.vertices[j]])
            except qhull.QhullError:
                self.mesh.centroids[j, :], self.mesh.volumes[j] = np.mean(self.points[self.mesh.vertices[j]]), 0

    def add_boundary_points_and_construct_delaunay(self):
        """
        This method add the corners of [0, 1]^dimension hypercube to the existing samples, which are used to construct a
        Delaunay Triangulation.
        """
        from scipy.spatial.qhull import Delaunay

        self.mesh_vertices = self.training_points.copy()
        self.points_to_samplesU01 = np.arange(0, self.training_points.shape[0])
        for i in range(np.shape(self.strata_object.voronoi.vertices)[0]):
            if any(np.logical_and(self.strata_object.voronoi.vertices[i, :] >= -1e-10,
                                  self.strata_object.voronoi.vertices[i, :] <= 1e-10)) or \
                    any(np.logical_and(self.strata_object.voronoi.vertices[i, :] >= 1 - 1e-10,
                                       self.strata_object.voronoi.vertices[i, :] <= 1 + 1e-10)):
                self.mesh_vertices = np.vstack(
                    [self.mesh_vertices, self.strata_object.voronoi.vertices[i, :]])
                self.points_to_samplesU01 = np.hstack([np.array([-1]), self.points_to_samplesU01, ])

        # Define the simplex mesh to be used for gradient estimation and sampling
        self.mesh = Delaunay(self.mesh_vertices, furthest_site=False, incremental=True, qhull_options=None)
        self.points = getattr(self.mesh, 'points')

    def calculate_strata_metrics(self):
        s = np.zeros(self.mesh.nsimplex)
        for j in range(self.mesh.nsimplex):
            s[j] = self.mesh.volumes[j] ** 2
        return s

    def update_strata_and_generate_samples(self, bins2break):
        new_points = np.zeros([p, self.dimension])
        for j in range(p):
            new_points[j, :] = self._generate_sample(bin2add[j])
        self._update_strata(new_point=new_points)
        return new_points

    def _update_strata(self, new_point):
        """
        This method update the `mesh` and `strata_object` attributes of refined_stratified_sampling class for each iteration.


        **Inputs:**

        * **new_point** (`ndarray`):
            An array of new samples generated at current iteration.
        """
        i_ = self.samples.shape[0]
        p_ = new_point.shape[0]
        # Update the matrices to have recognize the new point
        self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.arange(i_, i_ + p_)])
        self.mesh.old_vertices = self.mesh.vertices

        # Update the Delaunay triangulation mesh to include the new point.
        self.mesh.add_points(new_point)
        self.points = getattr(self.mesh, 'points')
        self.mesh_vertices = np.vstack([self.mesh_vertices, new_point])

        # Compute the strata weights.
        self.strata_object.voronoi, bounded_regions = Voronoi.voronoi_unit_hypercube(self.samplesU01)

        self.strata_object.centroids = []
        self.strata_object.volume = []
        for region in bounded_regions:
            vertices = self.strata_object.voronoi.vertices[region + [region[0]]]
            centroid, volume = Voronoi.compute_voronoi_centroid_volume(vertices)
            self.strata_object.centroids.append(centroid[0, :])
            self.strata_object.volume.append(volume)

    def _generate_sample(self, bin_):
        """
        This method create a subsimplex inside a Dealaunay Triangle and generate a random sample inside it using
        Simplex class.


        **Input:**

        * **bin_** (`int or float`):
            Index of delaunay triangle.


        **Outputt:**

        * **new** (`ndarray`):
            An array of new sample.

        """
        import itertools
        tmp_vertices = self.points[self.mesh.simplices[int(bin_), :]]
        col_one = np.array(list(itertools.combinations(np.arange(self.dimension + 1), self.dimension)))
        self.mesh.sub_simplex = np.zeros_like(tmp_vertices)  # node: an array containing mid-point of edges
        for m in range(self.dimension + 1):
            self.mesh.sub_simplex[m, :] = np.sum(tmp_vertices[col_one[m] - 1, :], 0) / self.dimension

        # Using the Simplex class to generate a new sample in the sub-simplex
        new = SimplexSampling(nodes=self.mesh.sub_simplex, samples_number=1, random_state=self.random_state).samples
        return new
