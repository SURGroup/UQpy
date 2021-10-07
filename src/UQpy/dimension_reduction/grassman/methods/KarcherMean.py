import numpy as np
import itertools


def karcher_mean(points_grassmann, p_planes_dimensions, optimization_method, distance):
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
        if p0 != p_planes_dimensions:
            raise ValueError('UQpy: The input points do not belong to the manifold G(n,p).')

    kr_mean = optimization_method.optimize(points_grassmann, distance)

    return kr_mean
