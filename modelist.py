import numpy as np
import scipy.integrate as integrate


########################################################################################################################
########################################################################################################################
#                                        List of models
########################################################################################################################

def model_zabaras(points, Type):
    d = 0.1
    return 1. / (abs(0.3 - np.sum(points ** 2, axis=0)) + d)


def model_runga(points):

    return 1. / (1 + 25 * points ** 2)


def model_sinusoidal(points):

    return np.sin(2.5 * points)


def model_wave1d(points):  # wave function
    t = 1
    d = 0.55

    if points < 0:
        y = np.sin(2 * np.pi * (0.5 * points * t + d))
    else:
        y = np.sin(np.pi * (0.5 * points * t + d))

    return y


def model_mike1d(points):
    """
    :param x:
    :return:
    """

    jump = 50
    c = 0.4
    order = 3
    if points < c:
        return points ** order
    else:
        return points ** order + jump


def model_ko1d(points, Type):
    """
    :param x:
    :return:
    """

    t = np.linspace(0, 10, 100)
    if points.size > 0:
        output = np.zeros(points.size)
        for i in range(points.size):
            xi1 = 1.0
            xi2 = points[0][i]
            xi3 = 0.0
            y0 = np.array([xi1, 0.1 * xi2, xi3])  # initials conditions
            Y = integrate.odeint(pend, y0, t, full_output=True)

            if Type == 'scalar':
                output[i] = Y[0][-1, 0]
            elif Type == 'vector':
                output[i] = Y[0][:, 0]
            else:
                raise NotImplementedError('K-O problem not supported for matrix output ')
    else:
        xi1 = 1.0
        xi2 = points
        xi3 = 0.0
        y0 = np.array([xi1, 0.1 * xi2, xi3])  # initials conditions
        Y = integrate.odeint(pend, y0, t, full_output=True)

        if Type == 'scalar':
            output = Y[0][-1, 0]
        elif Type == 'vector':
            output = Y[0][:, 0]
        else:
            raise NotImplementedError('K-O problem not supported for matrix output ')

    return output


def model_ko2d(x, Type):

    t = np.linspace(0, 10, 100)  # time
    xi1 = 1.0
    xi2 = x[0]
    xi3 = x[1]
    y0 = np.array([xi1, 0.1 * xi2, xi3])  # initials conditions
    solution = integrate.odeint(pend, y0, t, full_output=True)
    if Type == 'scalar':
        return solution[0][-1, 0]
    elif Type == 'vector':
        return solution[0][:, 0]


def model_ko3d(x):

    t = np.linspace(0, 10, 100)  # time
    xi1 = x[0]
    xi2 = x[1]
    xi3 = x[2]
    y0 = np.array([xi1, xi2, xi3])  # initials conditions
    solution = integrate.odeint(pend, y0, t, full_output=True)

    return solution


def pend(y, t=0):
    """
    :param y:
    :param t:
    :return:
    """
    return [y[0] * y[2], -y[1] * y[2], -y[0] ** 2 + y[1] ** 2]


def model_reliability(u, Type):
    a = 3
    return a - np.sum(u, axis=0) / np.sqrt(u.size)
