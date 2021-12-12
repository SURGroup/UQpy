import numpy as np


def series(z, k=7):
    t1 = 3 + 0.1 * (z[:, 1] - z[:, 0]) ** 2 - (z[:, 1] + z[:, 0]) / np.sqrt(2)
    t2 = 3 + 0.1 * (z[:, 1] - z[:, 0]) ** 2 + (z[:, 1] + z[:, 0]) / np.sqrt(2)
    t3 = z[:, 1] - z[:, 0] + k / np.sqrt(2)
    t4 = z[:, 0] - z[:, 1] + k / np.sqrt(2)
    return min([t1, t2, t3, t4])
