import copy
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.utilities.distances import GeodesicDistance
from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray, Numpy2DFloatArrayOrthonormal
from UQpy.utilities.distances.baseclass import GrassmannianDistance
from UQpy.utilities.kernels import GrassmannianKernel, ProjectionKernel


class GrassmannOperations:
    @beartype
    def __init__(self, grassmann_points: Union[list[Numpy2DFloatArrayOrthonormal], list[GrassmannPoint]],
                 kernel: GrassmannianKernel = ProjectionKernel(),
                 p: Union[int, str] = "max", optimization_method: str = "GradientDescent",
                 distance: GrassmannianDistance = GeodesicDistance()):
        """
        The :class:`.GrassmannOperations` class can used in two ways. In the first case, the user can invoke the
        initializer by providing all the required input data. The class will then automatically calculate the
        kernel matrix, distance matrix, karcher mean, and frechet variance of the input Grassmann points. Alternatively,
        the user may invoke each of the static methods individually and calculate only the required quantities, without
        having to instantiate a new object.

        :param grassmann_points: Data points projected on the Grassmann manifold
        :param kernel: Kernel to be used to evaluate the kernel matrix of the given Grassmann points
        :param p: Rank of the Grassmann projected points.
        :param optimization_method: String that defines the optimization method for computation of the Karcher mean.

            Options: `"GradientDescent"`, `"StochasticGradientDescent"`
        :param distance: Distance measure to be used for the optimization in computation of the Karcher mean and Frechet
            variance.
        """
        self.grassmann_points = grassmann_points
        self.p = p
        self.kernel_matrix = kernel.calculate_kernel_matrix(self.grassmann_points)
        self.distance_matrix = distance.calculate_distance_matrix(self.grassmann_points)
        self.karcher_mean = GrassmannOperations.karcher_mean(self.grassmann_points, optimization_method, distance)
        self.frechet_variance = GrassmannOperations.frechet_variance(self.grassmann_points, self.karcher_mean, distance)

    @staticmethod
    def calculate_kernel_matrix(grassmann_points: Union[list[Numpy2DFloatArrayOrthonormal], list[GrassmannPoint]],
                                kernel: GrassmannianKernel = ProjectionKernel()):
        return kernel.calculate_kernel_matrix(grassmann_points)

    @staticmethod
    @beartype
    def log_map(grassmann_points: Union[list[Numpy2DFloatArrayOrthonormal], list[GrassmannPoint]],
                reference_point: Union[Numpy2DFloatArrayOrthonormal, GrassmannPoint]) -> list[Numpy2DFloatArray]:
        """
        :param grassmann_points: Point(s) on the Grassmann manifold.
        :param reference_point: Origin of the tangent space.
        :return: Point(s) on the tangent space.
        """
        reference_point = reference_point.data if isinstance(reference_point, GrassmannPoint) else reference_point
        from UQpy.utilities.distances.baseclass.GrassmannianDistance import GrassmannianDistance
        number_of_points = len(grassmann_points)
        for i in range(number_of_points):
            GrassmannianDistance.check_rows(reference_point, grassmann_points[i])
            if reference_point.data.shape[1] != grassmann_points[i].data.shape[1]:
                raise ValueError("UQpy: Point {0} is on G({1},{2}) - Reference is on"
                                 " G({1},{2})".format(i, grassmann_points[i].data.shape[1],
                                                      grassmann_points[i].data.shape[0]))

        # Multiply ref by its transpose.
        reference_point_transpose = reference_point.T
        m_ = np.dot(reference_point.data, reference_point_transpose)

        tangent_points = []
        for i in range(number_of_points):
            u_trunc = grassmann_points[i]
            # compute: M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
            m_inv = np.linalg.inv(np.dot(reference_point_transpose, u_trunc.data))
            m = np.dot(u_trunc.data - np.dot(m_, u_trunc.data), m_inv)
            ui, si, vi = np.linalg.svd(m, full_matrices=False)
            tangent_points.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        return tangent_points

    @staticmethod
    @beartype
    def exp_map(tangent_points: list[Numpy2DFloatArray],
                reference_point: Union[np.ndarray, GrassmannPoint]) -> list[GrassmannPoint]:
        """
        :param tangent_points: Tangent vector(s).
        :param reference_point: Origin of the tangent space.
        :return: Point(s) on the Grassmann manifold.
        """
        number_of_points = len(tangent_points)
        for i in range(number_of_points):
            if reference_point.data.shape[1] != tangent_points[i].shape[1]:
                raise ValueError("UQpy: Point {0} is on G({1},{2}) - Reference is on"
                                 " G({1},{2})".format(i, tangent_points[i].shape[1], tangent_points[i].shape[0]))

        # Map the each point back to the manifold.
        manifold_points = []
        for i in range(number_of_points):
            u_trunc = tangent_points[i]
            ui, si, vi = np.linalg.svd(u_trunc, full_matrices=False)

            x0 = np.dot(
                np.dot(np.dot(reference_point.data, vi.T), np.diag(np.cos(si)))
                + np.dot(ui, np.diag(np.sin(si))),
                vi,
            )

            if not np.allclose(x0.T @ x0, np.eye(u_trunc.shape[1])):
                x0, _ = np.linalg.qr(x0)

            manifold_points.append(GrassmannPoint(x0))

        return manifold_points

    @staticmethod
    @beartype
    def frechet_variance(grassmann_points: Union[list[Numpy2DFloatArrayOrthonormal], list[GrassmannPoint]],
                         reference_point: Union[Numpy2DFloatArrayOrthonormal, GrassmannPoint],
                         distance: GrassmannianDistance) -> float:
        """
        :param grassmann_points: Point(s) on the Grassmann manifold
        :param reference_point: Reference point for the Frechet variance (:math:`\mu`). Typically assigned as the
            Karcher mean.
        :param distance: Distance measure to be used in the variance calculation.
        """
        p_dim = [min(np.shape(grassmann_points[i].data)) for i in range(len(grassmann_points))]

        points_number = len(grassmann_points)

        variance_nominator = 0
        for i in range(points_number):
            distance.calculate_distance_matrix([reference_point, grassmann_points[i]], p_dim)
            distances = distance.distance_matrix
            variance_nominator += distances[0] ** 2

        return variance_nominator / points_number

    @staticmethod
    @beartype
    def karcher_mean(grassmann_points: Union[list[Numpy2DFloatArrayOrthonormal], list[GrassmannPoint]],
                     optimization_method: str, distance: GrassmannianDistance,
                     acceleration: bool = False, tolerance: float = 1e-3,
                     maximum_iterations: int = 1000) -> GrassmannPoint:
        """
        :param maximum_iterations: Maximum number of iterations performed by the optimization algorithm.
        :param tolerance: Tolerance used as the convergence criterion of the optimization.
        :param acceleration: Boolean flag used in combination with :code:`GradientDescent` optimization method which
         activates the Nesterov acceleration scheme
        :param grassmann_points: Point(s) on the Grassmann manifold.
        :param optimization_method: String that defines the optimization method.

            Options: `"GradientDescent"`, `"StochasticGradientDescent"`
        :param distance: Distance measure to be used for the optimization.
        """
        # Compute and test the number of input matrices necessary to compute the Karcher mean.
        nargs = len(grassmann_points)
        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")

        if optimization_method == "GradientDescent":
            return GrassmannOperations._gradient_descent(grassmann_points, distance, acceleration, tolerance, maximum_iterations)
        else:
            return GrassmannOperations._stochastic_gradient_descent(grassmann_points, distance, tolerance, maximum_iterations)

    @staticmethod
    def _gradient_descent(data_points, distance_fun, acceleration, tolerance, maximum_iterations):
        # acc is a boolean variable to activate the Nesterov acceleration scheme.
        acc = acceleration
        # Error tolerance
        tol = tolerance
        # Maximum number of iterations.
        maxiter = maximum_iterations
        # Number of points.
        n_mat = len(data_points)

        # =========================================
        alpha = 0.5
        rnk = [min(np.shape(data_points[i].data)) for i in range(n_mat)]
        max_rank = max(rnk)
        fmean = [GrassmannOperations.frechet_variance(data_points, data_points[i], distance_fun) for i in range(n_mat)]

        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].data.tolist()

        avg_gamma = np.zeros([np.shape(data_points[0].data)[0], np.shape(data_points[0].data)[1]])

        itera = 0

        l = 0
        avg = []
        _gamma = []
        if acc:
            _gamma = GrassmannOperations.log_map(grassmann_points=data_points,
                                                 reference_point=np.asarray(mean_element))

            avg_gamma.fill(0)
            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat
            avg.append(avg_gamma)

        # Main loop
        while itera <= maxiter:
            _gamma = GrassmannOperations.log_map(grassmann_points=data_points,
                                                 reference_point=np.asarray(mean_element))
            avg_gamma.fill(0)

            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat

            test_0 = np.linalg.norm(avg_gamma, 'fro')
            if test_0 < tol and itera == 0:
                break

            # Nesterov: Accelerated Gradient Descent
            if acc:
                avg.append(avg_gamma)
                l0 = l
                l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                ls = (1 - l0) / l1
                step = (1 - ls) * avg[itera + 1] + ls * avg[itera]
                l = copy.copy(l1)
            else:
                step = alpha * avg_gamma

            x = GrassmannOperations.exp_map(tangent_points=[step],
                                            reference_point=np.asarray(mean_element))

            test_1 = np.linalg.norm(x[0].data - mean_element, 'fro')

            if test_1 < tol:
                break

            mean_element = []
            mean_element = x[0].data.tolist()

            itera += 1

        # return the Karcher mean.
        return GrassmannPoint(np.asarray(mean_element))

    @staticmethod
    def _stochastic_gradient_descent(data_points, distance_fun, tolerance, maximum_iterations):

        tol = tolerance
        maxiter = maximum_iterations
        n_mat = len(data_points)

        rnk = [min(np.shape(data_points[i].data)) for i in range(n_mat)]
        max_rank = max(rnk)

        fmean = [GrassmannOperations.frechet_variance(data_points, data_points[i], distance_fun) for i in range(n_mat)]

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].data.tolist()
        itera = 0
        _gamma = []
        k = 1
        while itera < maxiter:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = GrassmannOperations.log_map(grassmann_points=[data_points[idx]],
                                                     reference_point=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = GrassmannOperations.exp_map(tangent_points=[step],
                                                reference_point=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0].data

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        return GrassmannPoint(np.asarray(mean_element))
