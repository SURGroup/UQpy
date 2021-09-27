import numpy as np
import itertools


class KarcherMean:

    def __init__(self, distance, optimization_method, p_planes_dimensions):
        self.p_planes_dimensions = p_planes_dimensions
        self.optimization_method = optimization_method
        self.distance = distance


    def compute_mean(self, points_grassmann):

        """
        Compute the Karcher mean.

        This method computes the Karcher mean given a set of points on the grassman manifold. The Karcher mean is
        estimated by the minimization of the Frechet variance, where the Frechet variance corresponds to the sum of the
        square distances, defined on the grassman manifold, to a given point. The command to compute the Karcher mean
        given a seto of points on the grassman manifold is.

        In this case two values are returned corresponding to the ones related to the manifolds defined by the left and
        right singular eigenvectors.

        **Input:**

        * **points_grassmann** (`list` or `ndarray`)
            Matrices (at least 2) corresponding to the points on the grassman manifold.

        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean. If ``gradient_descent`` is
            employed the keywords are `acc`, a boolean variable for the accelerated method; `tol`, tolerance with
            default value equal to 1e-3; and `maxiter`, maximum number of iterations with default value equal to 1000.
            If `stochastic_gradient_descent` is employed instead, `acc` is not used.

        **Output/Returns:**

        * **kr_mean** (`list`)
            Karcher mean.

        * **kr_mean_psi** (`list`)
            Karcher mean for left singular eigenvectors if `points_grassmann` is not provided.

        * **kr_mean_phi** (`list`)
            Karcher mean for right singular eigenvectors if `points_grassmann` is not provided.

        """

        # Test the input data for type consistency.
        if not isinstance(points_grassmann, list) and not isinstance(points_grassmann, np.ndarray):
            raise TypeError('UQpy: `points_grassmann` must be either list or numpy.ndarray.')

        # Compute and test the number of input matrices necessary to compute the Karcher mean.
        nargs = len(points_grassmann)
        if nargs < 2:
            raise ValueError('UQpy: At least two matrices must be provided.')

        # Test the dimensionality of the input data.
        p = []
        for i in range(len(points_grassmann)):
            p.append(min(np.shape(np.array(points_grassmann[i]))))

        if p.count(p[0]) != len(p):
            raise TypeError('UQpy: The input points do not belong to the same manifold.')
        else:
            p0 = p[0]
            if p0 != self.p_planes_dimensions:
                raise ValueError('UQpy: The input points do not belong to the manifold G(n,p).')

        kr_mean = self.optimization_method.optimize(points_grassmann, self.distance)

        return kr_mean

    @staticmethod
    def frechet_variance(reference_point, points_grassmann, distance):
        p_dim = []
        for i in range(len(points_grassmann)):
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

        points_number = len(points_grassmann)

        if points_number < 2:
            raise ValueError('UQpy: At least two input matrices must be provided.')

        variance_nominator = 0
        for i in range(points_number):
            distances = KarcherMean.__estimate_distance([reference_point, points_grassmann[i]], p_dim, distance)
            variance_nominator += distances[0] ** 2

        frechet_variance = variance_nominator / points_number
        return frechet_variance

    @staticmethod
    def __estimate_distance(points, p_dim, distance):
        """
        Private method: Estimate the distance between points on the grassman manifold.

        This is an auxiliary method to compute the pairwise distance of points on the grassman manifold.
        The input arguments are passed through a list . Further, the user has the option to pass the dimension
        of the embedding space.

        **Input:**

        * **points** (`list` or `ndarray`)
            Matrices (at least 2) corresponding to the points on the grassman manifold.

        * **p_dim** (`list`)
            Embedding dimension.

        **Output/Returns:**

        * **distance_list** (`list`)
            Pairwise distance.

        """

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError('UQpy: The input matrices must be either list or numpy.ndarray.')

        nargs = len(points)

        if nargs < 2:
            raise ValueError('UQpy: At least two matrices must be provided.')

        # ------------------------------------------------------------

        # Define the pairs of points to compute the grassman distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Compute the pairwise distances.
        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            p0 = int(p_dim[ii])
            p1 = int(p_dim[jj])

            x0 = np.asarray(points[ii])[:, :p0]
            x1 = np.asarray(points[jj])[:, :p1]

            # Call the functions where the distance metric is implemented.
            distance_value = distance.compute_distance(x0, x1)

            distance_list.append(distance_value)

        return distance_list