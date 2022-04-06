import copy

import numpy as np
import numpy.random
from beartype import beartype

from UQpy.utilities.distances.grassmannian_distances import GeodesicDistance
from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.SVDProjection import SVDProjection
from UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations import GrassmannOperations
import sys
from numpy.random import RandomState

sol0 = np.array([[0.61415,1.03029,1.02001,0.57327,0.79874,0.73274],
                    [0.56924,0.91700,0.88841,0.53737,0.68676,0.67751],
                    [0.51514,0.87898,0.87779,0.47850,0.69085,0.61525],
                    [0.63038,1.10822,1.12313,0.58038,0.89142,0.75429],
                    [0.69666,1.03114,0.95037,0.67211,0.71184,0.82522],
                    [0.66595,1.03789,0.98690,0.63420,0.75416,0.79110]])

sol1 = np.array([[1.05134,1.37652,0.95634,0.85630,0.47570,1.22488],
                [0.16370,0.63105,0.14533,0.81030,0.44559,0.43358],
                [1.23478,2.10342,1.04698,1.68755,0.92792,1.73277],
                [0.90538,1.64067,0.62027,1.17577,0.63644,1.34925],
                [0.58210,0.75795,0.65519,0.65712,0.37251,0.65740],
                [0.99174,1.59375,0.63724,0.89107,0.47631,1.36581]])

sol2 = np.array([[1.04142,0.91670,1.47962,1.23350,0.94111,0.61858],
                [1.00464,0.65684,1.35136,1.11288,0.96093,0.42340],
                [1.05567,1.33192,1.56286,1.43412,0.77044,0.97182],
                [0.89812,0.86136,1.20204,1.17892,0.83788,0.61160],
                [0.46935,0.39371,0.63534,0.57856,0.47615,0.26407],
                [1.14102,0.80869,1.39123,1.33076,0.47719,0.68170]])

sol3 = np.array([[0.60547,0.11492,0.78956,0.13796,0.76685,0.41661],
                [0.32771,0.11606,0.67630,0.15208,0.44845,0.34840],
                [0.58959,0.10156,0.72623,0.11859,0.73671,0.38714],
                [0.36283,0.07979,0.52824,0.09760,0.46313,0.27906],
                [0.87487,0.22452,1.30208,0.30189,1.22015,0.62918],
                [0.56006,0.16879,1.09635,0.20431,0.69439,0.60317]])

def test_karcher():
    numpy.random.seed(0)
    rnd = RandomState(0)
    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SVDProjection(matrices, p="max")

    # optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    psi_mean = GrassmannOperations.karcher_mean(grassmann_points=manifold_projection.u,
                                                optimization_method="GradientDescent",
                                                distance=GeodesicDistance())

    phi_mean = GrassmannOperations.karcher_mean(grassmann_points=manifold_projection.v,
                                                optimization_method="GradientDescent",
                                                distance=GeodesicDistance())

    assert round(psi_mean.data[0, 0], 9) == -0.418075902
    assert round(phi_mean.data[0, 0], 9) == -0.353239531

def test_karcher_stochastic():
    np.random.seed(1111)
    rnd = RandomState(0)
    # Solutions: original space.

    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SVDProjection(matrices, p="max")

    # optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    psi_mean = GrassmannOperations.karcher_mean(grassmann_points=manifold_projection.u,
                                                optimization_method="StochasticGradientDescent",
                                                distance=GeodesicDistance())

    phi_mean = GrassmannOperations.karcher_mean(grassmann_points=manifold_projection.v,
                                                optimization_method="StochasticGradientDescent",
                                                distance=GeodesicDistance())

    assert round(psi_mean.data[0, 0], 9) == -0.418075902
    assert round(phi_mean.data[0, 0], 9) == -0.353239531