import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.GrassmanDistance import GrassmannDistance
from UQpy.dimension_reduction.grassman.KarcherMean import KarcherMean
from UQpy.dimension_reduction.grassman.interpolations.LinearInterpolation import LinearInterpolation
from UQpy.dimension_reduction.grassman.manifold_projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent
from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel


class Grassmann:

    def __init__(self, manifold_projected_points: ManifoldProjection):
        self.manifold_projected_points = manifold_projected_points

    def interpolate(self, coordinates, point, element_wise=True,
                    karcher_mean: KarcherMean =None,
                    interpolation_method=None):
        if karcher_mean is None:
            karcher_mean = KarcherMean(distance=GrassmannDistance(), optimization_method=GradientDescent(),
                                       p_planes_dimensions=self.manifold_projected_points.p_planes_dimensions)
        if interpolation_method is None:
            interpolation_method = LinearInterpolation()

        return self.manifold_projected_points.interpolate(karcher_mean, interpolation_method,
                                                          coordinates, point, element_wise)

    def evaluate_kernel_matrix(self, kernel):
        kernel_matrix = self.manifold_projected_points.evaluate_matrix(kernel)
        return kernel_matrix

    @staticmethod
    def log_map(points_grassmann=None, reference_point=None):
        # Show an error message if points_grassmann is not provided.
        if points_grassmann is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if reference_point is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Check points_grassmann for type consistency.
        if not isinstance(points_grassmann, list) and not isinstance(points_grassmann, np.ndarray):
            raise TypeError('UQpy: `points_grassmann` must be either a list or numpy.ndarray.')

        # Get the number of matrices in the set.
        points_number = len(points_grassmann)

        shape_0 = np.shape(points_grassmann[0])
        shape_ref = np.shape(reference_point)
        p_dim = []
        for i in range(points_number):
            shape = np.shape(points_grassmann[i])
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # Check reference for type consistency.
        reference_point = np.asarray(reference_point)
        if not isinstance(reference_point, list):
            ref_list = reference_point.tolist()
        else:
            ref_list = reference_point
            reference_point = np.array(reference_point)

        # Multiply ref by its transpose.
        refT = reference_point.T
        m0 = np.dot(reference_point, refT)

        # Loop over all the input matrices.
        tangent_points = []
        for i in range(points_number):
            utrunc = points_grassmann[i][:, 0:p0]

            # If the reference point is one of the given points
            # set the entries to zero.
            if utrunc.tolist() == ref_list:
                tangent_points.append(np.zeros(np.shape(reference_point)))
            else:
                # compute: M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
                minv = np.linalg.inv(np.dot(refT, utrunc))
                m = np.dot(utrunc - np.dot(m0, utrunc), minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)  # svd(m, max_rank)
                tangent_points.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        # Return the points on the tangent space
        return tangent_points

    @staticmethod
    def exp_map(points_tangent=None, reference_point=None):

        """
        Map points on the tangent space onto the Grassmann manifold.

        It maps the points on the tangent space, passed to the method using points_tangent, onto the Grassmann manifold.
        It is mandatory that the user pass a reference point where the tangent space was created.

        **Input:**

        * **points_tangent** (`list`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.

        * **ref** (`list` or `ndarray`)
            A point on the Grassmann manifold used as reference to construct the tangent space.

        **Output/Returns:**

        * **points_manifold**: (`list`)
            Point on the tangent space.

        """

        # Show an error message if points_tangent is not provided.
        if points_tangent is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if reference_point is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Test points_tangent for type consistency.
        if not isinstance(points_tangent, list) and not isinstance(points_tangent, np.ndarray):
            raise TypeError('UQpy: `points_tangent` must be either list or numpy.ndarray.')

        # Number of input matrices.
        nargs = len(points_tangent)

        shape_0 = np.shape(points_tangent[0])
        shape_ref = np.shape(reference_point)
        p_dim = []
        for i in range(nargs):
            shape = np.shape(points_tangent[i])
            p_dim.append(min(np.shape(np.array(points_tangent[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # -----------------------------------------------------------

        reference_point = np.array(reference_point)

        # Map the each point back to the manifold.
        points_manifold = []
        for i in range(nargs):
            utrunc = points_tangent[i][:, :p0]
            ui, si, vi = np.linalg.svd(utrunc, full_matrices=False)

            # Exponential mapping.
            x0 = np.dot(np.dot(np.dot(reference_point, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality.
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing.

            points_manifold.append(x0)

        return points_manifold

