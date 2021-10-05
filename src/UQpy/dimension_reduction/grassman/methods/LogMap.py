import numpy as np


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
