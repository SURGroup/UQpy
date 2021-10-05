import numpy as np


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
