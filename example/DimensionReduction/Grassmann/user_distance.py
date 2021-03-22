import numpy as np

def my_distance(x0, x1):

    """
    Estimate the user distance.

    **Input:**

    * **x0** (`list` or `ndarray`)
        Point on the Grassmann manifold.

    * **x1** (`list` or `ndarray`)
        Point on the Grassmann manifold.

    **Output/Returns:**

    * **distance** (`float`)
        Procrustes distance between x0 and x1.
    """

    if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
        raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
    else:
        x0 = np.array(x0)

    if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
        raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
    else:
        x1 = np.array(x1)

    l = min(np.shape(x0))
    k = min(np.shape(x1))
    rank = min(l, k)

    r = np.dot(x0.T, x1)
    #(ui, si, vi) = svd(r, rank)
    
    ui, si, vi = np.linalg.svd(r, full_matrices=True,hermitian=False)  # Compute the SVD of matrix
    si = np.diag(si)  # Transform the array si into a diagonal matrix containing the singular values
    vi = vi.T  # Transpose of vi

    u = ui[:, :rank]
    s = si[:rank, :rank]
    v = vi[:, :rank]
    
    index = np.where(si > 1)
    si[index] = 1.0
    theta = np.arccos(si)
    theta = np.sin(theta / 2) ** 2
    distance = np.sqrt(abs(k - l) + 2 * np.sum(theta))

    return distance