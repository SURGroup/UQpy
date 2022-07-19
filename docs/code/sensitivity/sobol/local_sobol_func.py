"""

Auxiliary file
==============================================

"""

import numpy as np
import copy


def evaluate(X, a_values):

    dims = len(a_values)
    g = 1

    for i in range(dims):
        g_i = (np.abs(4 * X[:, i] - 2) + a_values[i]) / (1 + a_values[i])
        g *= g_i

    return g


def sensitivities(a_values):

    dims = len(a_values)

    Total_order = np.zeros((dims, 1))

    V_i = (3 * (1 + a_values) ** 2) ** (-1)

    total_variance = np.prod(1 + V_i) - 1

    First_order = V_i / total_variance

    for i in range(dims):

        rem_First_order = copy.deepcopy(V_i)
        rem_First_order[i] = 0
        Total_order[i] = V_i[i] * np.prod(rem_First_order + 1) / total_variance

    return First_order.reshape(-1, 1), Total_order
