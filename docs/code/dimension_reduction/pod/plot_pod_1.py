"""

POD on data
=====================

In this example, the Direct Proper Orthogonal Decomposition (POD) method is used to decompose a set of data and extract
basis functions which can be used for the reconstruction of the solution.
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy, matplotlib, and we also import the
# POD class from UQpy.

# %%

from UQpy.dimension_reduction import DirectPOD
import time
import numpy as np

# %% md
#
# Input dataset in the form of a second order tensor consists of three snapshot solutions. Using the POD method we
# reconstruct the input dataset by keeping a predefined number of modes.

# %%

Data = np.zeros((3, 5, 3))

Data[:, :, 0] = [
    [0.9073, 1.7842, 2.1236, 1.1323, 1.6545],
    [0.8924, 1.7753, -0.6631, 0.5654, 2.1235],
    [2.1488, 4.2495, 1.8260, 0.3423, 4.9801]]

Data[:, :, 1] = [
    [0.7158, 1.6970, -0.0740, 5.478, 1.0987],
    [-0.4898, -1.5077, 1.9103, 6.7121, 0.5334],
    [0.3054, 0.3207, 2.1335, 1.1082, 5.5435]]

Data[:, :, 2] = [
    [-0.3698, 0.0151, 1.4429, 2.5463, 6.9871],
    [2.4288, 4.0337, -1.7495, -0.5012, 1.7654],
    [2.3753, 4.7146, -0.2716, 1.6543, 0.9121]]

# %% md
#
# The Direct POD method is used to compute the reconstructed solutions and reduced-order solutions in the spatial
# dimension of the data. Full reconstruction is achieved when the number of modes chosen, equals the number of
# dimensions.

# %%

start_time = time.time()

pod = DirectPOD(solution_snapshots=Data, modes=1)
Data_reconstr = pod.run()

# %% md
#
# Print the reconstructed dataset.

# %%

print('Reconstructed snapshot no.1:')
print(Data_reconstr[0][:, :, 0])

if np.allclose(Data, Data_reconstr[0]) == True:
    print('Input data and reconstructed data are identical.')

elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('Elapsed time: ', round(elapsed_time, 8))
