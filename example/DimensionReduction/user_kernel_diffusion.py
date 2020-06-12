import numpy as np
import itertools
import scipy.spatial.distance as sd

def my_kernel_diffusion(data):

    epsilon=0.3
    # Compute the pairwise distances.
    if len(np.shape(data)) == 2:
        # Set of 1-D arrays
        distance_pairs = sd.pdist(data, 'euclidean')
    elif len(np.shape(data)) == 3:
        # Set of 2-D arrays
        # Check arguments: verify the consistency of input arguments.
        nargs = len(data)
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        distance_pairs = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = data[ii]
            x1 = data[jj]

            distance = np.linalg.norm(x0 - x1, 'fro')

            distance_pairs.append(distance)
    else:
        raise TypeError('UQpy: The size of the input data is not consistent with this method.')
        
    kernel_matrix = np.exp(-sd.squareform(distance_pairs) ** 2 / (4 * epsilon))
    
    return kernel_matrix