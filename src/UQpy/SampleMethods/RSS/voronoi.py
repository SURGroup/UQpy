import numpy as np

from UQpy.SampleMethods.RSS.rss import RSS
from UQpy.SampleMethods.Simplex import Simplex
from UQpy.SampleMethods.Strata import VoronoiStrata, DelaunayStrata


class VoronoiRSS(RSS):
    """
    Executes Refined Stratified Sampling using Voronoi Stratification.

    ``VoronoiRSS`` is a child class of ``RSS``. ``VoronoiRSS`` takes in all parameters defined in the parent
    ``RSS`` class with differences note below. Only those inputs and attributes that differ from the parent class
    are listed below. See documentation for ``RSS`` for additional details.

    **Inputs:**

    * **sample_object** (``SampleMethods`` object):
        The `sample_object` for ``VoronoiRSS`` can be an object of any ``SampleMethods`` class that possesses the
        following attributes: `samples` and `samplesU01`

        This can be any ``SampleMethods`` object because ``VoronoiRSS`` creates its own `strata_object`. It does not use
        a `strata_object` inherited from an ``STS`` object.

    **Methods:**
    """

    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        if hasattr(sample_object, 'samplesU01'):
            self.strata_object = VoronoiStrata(seeds=sample_object.samplesU01)

        self.mesh = None
        self.mesh_vertices, self.vertices_in_U01 = [], []
        self.points_to_samplesU01, self.points = [], []

        super().__init__(sample_object=sample_object, runmodel_object=runmodel_object, krig_object=krig_object,
                         local=local, max_train_size=max_train_size, step_size=step_size, qoi_name=qoi_name,
                         n_add=n_add, nsamples=nsamples, random_state=random_state, verbose=verbose)

    def run_rss(self):
        """
        Overwrites the ``run_rss`` method in the parent class to perform refined stratified sampling with Voronoi
        strata. It is an instance method that does not take any additional input arguments. See
        the ``RSS`` class for additional details.
        """
        if self.runmodel_object is not None:
            self._gerss()
        else:
            self._rss()

        self.weights = self.strata_object.volume

    def _gerss(self):
        """
        This method generates samples using Gradient Enhanced Refined Stratified Sampling.
        """
        import math

        # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
        self._add_boundary_points_and_construct_delaunay()

        self.mesh.old_vertices = self.mesh.vertices

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # Compute the centroids and the volumes of each simplex cell in the mesh
            self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
            self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
            from scipy.spatial import qhull, ConvexHull
            for j in range(self.mesh.nsimplex):
                try:
                    ConvexHull(self.points[self.mesh.vertices[j]])
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        DelaunayStrata.compute_delaunay_centroid_volume(self.points[self.mesh.vertices[j]])
                except qhull.QhullError:
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = np.mean(self.points[self.mesh.vertices[j]]), 0

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                qoi = self.runmodel_object.qoi_list

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the gradients at the existing sample points
            if self.max_train_size is None or len(self.training_points) <= self.max_train_size or \
                    i == self.samples.shape[0]:
                # Use the entire sample set to train the surrogate model (more expensive option)
                dy_dx = self.estimate_gradient(np.atleast_2d(self.training_points), qoi, self.mesh.centroids)
            else:
                # Use only max_train_size points to train the surrogate model (more economical option)
                # Build a mapping from the new vertex indices to the old vertex indices.
                self.mesh.new_vertices, self.mesh.new_indices = [], []
                self.mesh.new_to_old = np.zeros([self.mesh.vertices.shape[0], ]) * np.nan
                j, k = 0, 0
                while j < self.mesh.vertices.shape[0] and k < self.mesh.old_vertices.shape[0]:

                    if np.all(self.mesh.vertices[j, :] == self.mesh.old_vertices[k, :]):
                        self.mesh.new_to_old[j] = int(k)
                        j += 1
                        k = 0
                    else:
                        k += 1
                        if k == self.mesh.old_vertices.shape[0]:
                            self.mesh.new_vertices.append(self.mesh.vertices[j])
                            self.mesh.new_indices.append(j)
                            j += 1
                            k = 0

                # Find the nearest neighbors to the most recently added point
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=self.max_train_size)
                knn.fit(np.atleast_2d(self.samplesU01))
                neighbors = knn.kneighbors(np.atleast_2d(self.samplesU01[-1]), return_distance=False)

                # For every simplex, check if at least dimension-1 vertices are in the neighbor set.
                # Only update the gradient in simplices that meet this criterion.
                update_list = []
                for j in range(self.mesh.vertices.shape[0]):
                    self.vertices_in_U01 = self.points_to_samplesU01[self.mesh.vertices[j]]
                    self.vertices_in_U01[np.isnan(self.vertices_in_U01)] = 10 ** 18
                    v_set = set(self.vertices_in_U01)
                    v_list = list(self.vertices_in_U01)
                    if len(v_set) != len(v_list):
                        continue
                    else:
                        if all(np.isin(self.vertices_in_U01, np.hstack([neighbors, np.atleast_2d(10 ** 18)]))):
                            update_list.append(j)

                update_array = np.asarray(update_list)

                # Initialize the gradient vector
                dy_dx = np.zeros((self.mesh.new_to_old.shape[0], self.dimension))

                # For those simplices that will not be updated, use the previous gradient
                for j in range(dy_dx.shape[0]):
                    if np.isnan(self.mesh.new_to_old[j]):
                        continue
                    else:
                        dy_dx[j, :] = dy_dx_old[int(self.mesh.new_to_old[j]), :]

                # For those simplices that will be updated, compute the new gradient
                dy_dx[update_array, :] = self.estimate_gradient(np.squeeze(self.samplesU01[neighbors]),
                                                                np.atleast_2d(np.array(qoi)[neighbors]),
                                                                self.mesh.centroids[update_array])

            # Determine the simplex to break and draw a new sample

            # Estimate the variance over each simplex by Delta Method. Moments of the simplices are computed using
            # Eq. (19) from the following reference:
            # Good, I.J. and Gaskins, R.A. (1971). The Centroid Method of Numerical Integration. Numerische
            #       Mathematik. 16: 343--359.
            var = np.zeros((self.mesh.nsimplex, self.dimension))
            s = np.zeros(self.mesh.nsimplex)
            for j in range(self.mesh.nsimplex):
                for k in range(self.dimension):
                    std = np.std(self.points[self.mesh.vertices[j]][:, k])
                    var[j, k] = (self.mesh.volumes[j] * math.factorial(self.dimension) /
                                 math.factorial(self.dimension + 2)) * (self.dimension * std ** 2)
                s[j] = np.sum(dy_dx[j, :] * var[j, :] * dy_dx[j, :]) * (self.mesh.volumes[j] ** 2)
            dy_dx_old = dy_dx

            # 'p' is number of samples to be added in the current iteration
            bin2add = self.identify_bins(strata_metric=s, p_=p)

            # Create 'p' sub-simplex within the simplex with maximum variance
            new_points = np.zeros([p, self.dimension])
            for j in range(p):
                new_points[j, :] = self._generate_sample(bin2add[j])

            # ###########################
            # ---------------------------
            # 2. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###########################
            # ---------------------------
            # 3. Update strata attributes
            # ---------------------------
            self._update_strata(new_point=new_points)

            # ###############################
            # -------------------------------
            # 4. Execute model at new samples
            # -------------------------------
            self.runmodel_object.run(samples=self.samples[-self.n_add:])

            if self.verbose:
                print("Iteration:", i)

    def _rss(self):
        """
        This method generates samples using Refined Stratified Sampling.
        """

        # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
        self._add_boundary_points_and_construct_delaunay()

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the centroids and the volumes of each simplex cell in the mesh
            self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
            self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
            from scipy.spatial import qhull, ConvexHull
            for j in range(self.mesh.nsimplex):
                try:
                    ConvexHull(self.points[self.mesh.vertices[j]])
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        DelaunayStrata.compute_delaunay_centroid_volume(self.points[self.mesh.vertices[j]])
                except qhull.QhullError:
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = np.mean(self.points[self.mesh.vertices[j]]), 0

            # Determine the simplex to break and draw a new sample
            s = np.zeros(self.mesh.nsimplex)
            for j in range(self.mesh.nsimplex):
                s[j] = self.mesh.volumes[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2add = self.identify_bins(strata_metric=s, p_=p)

            # Create 'p' sub-simplex within the simplex with maximum variance
            new_points = np.zeros([p, self.dimension])
            for j in range(p):
                new_points[j, :] = self._generate_sample(bin2add[j])

            # ###########################
            # ---------------------------
            # 2. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###########################
            # ---------------------------
            # 3. Update strata attributes
            # ---------------------------
            self._update_strata(new_point=new_points)

            if self.verbose:
                print("Iteration:", i)

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
        new = Simplex(nodes=self.mesh.sub_simplex, nsamples=1, random_state=self.random_state).samples
        return new

    def _update_strata(self, new_point):
        """
        This method update the `mesh` and `strata_object` attributes of RSS class for each iteration.


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
        self.strata_object.voronoi, bounded_regions = VoronoiStrata.voronoi_unit_hypercube(self.samplesU01)

        self.strata_object.centroids = []
        self.strata_object.volume = []
        for region in bounded_regions:
            vertices = self.strata_object.voronoi.vertices[region + [region[0]]]
            centroid, volume = VoronoiStrata.compute_voronoi_centroid_volume(vertices)
            self.strata_object.centroids.append(centroid[0, :])
            self.strata_object.volume.append(volume)

    def _add_boundary_points_and_construct_delaunay(self):
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