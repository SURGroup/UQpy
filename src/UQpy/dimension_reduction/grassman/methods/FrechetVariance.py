import itertools

import numpy as np

def frechet_variance(reference_point, points_grassmann, distance):
    p_dim = []
    for i in range(len(points_grassmann)):
        p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

    points_number = len(points_grassmann)

    if points_number < 2:
        raise ValueError('UQpy: At least two input matrices must be provided.')

    variance_nominator = 0
    for i in range(points_number):
        distances = __estimate_distance([reference_point, points_grassmann[i]], p_dim, distance)
        variance_nominator += distances[0] ** 2

    frechet_variance = variance_nominator / points_number
    return frechet_variance


def __estimate_distance(points, p_dim, distance):

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
