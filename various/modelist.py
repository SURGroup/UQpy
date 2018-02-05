import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats


########################################################################################################################
########################################################################################################################
#                                        List of models
########################################################################################################################

def model_zabaras(points, Type):
    """
    Analytical function defined in :math:`[0,1]^2`.

    :math:`model\_func1 = \\frac{1}{0.3 - \\Sigma{{points}^2} + 0.1}`


    :param points: Sample point to evaluate the model
    :return: The value of the model at the sample point

    Reference: http://dx.doi.org/10.1016/j.jcp.2009.01.006

    """

    return 1. / (abs(0.3 - np.sum(points ** 2, axis=0)) + 0.01)


def model_runge(points):
    """
    Analytical function defined in :math:`[-1, 1]`.


    :param points: Sample point to evaluate the model
    :return: The value of the model at the sample point
    """

    return 1. / (1 + 25 * points ** 2)


def model_mike1d(points):
    """
    One-dimensional Analytical function defined in :math:`[-1, 1]`.


    :param points: Sample point to evaluate the model
    :return: The value of the model at the sample point
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
    Kraichnan–Orszag (K–O) 3-mode problem with one random variable defined in :math:`[-1, 1]`.


    :param points:Sample point to evaluate the model
    :param Type: (Optional)Type of the output:
                 1. Scalar
                 2. Vector
                 3. Tensor
            For this example the only option is Scalar
    :return: The value of the model at the sample point
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


def model_eigenvalues(X):
    coeff = [-1, X[0] + 2 * X[1], -(X[0] * X[1] + 2 * X[0] * X[2] + 3 * X[1] * X[2] + X[2] ** 2),
             (X[0] * X[1] * X[2] + (X[0] + X[1]) * X[2])]
    return np.roots(coeff)
########################################################################################################################
########################################################################################################################
#                                        List of distribution
########################################################################################################################


def normpdf(x):
    """ Normal density function used to generate samples using Metropolis-Hastings Algorithm
     :math: `f(x) = \\frac{1}{(2*\\pi*\\sigma)^(1/2)}*exp(-\\frac{1}{2}*(\\frac{x-\\mu}{\\sigma})^2)`

    """
    return stats.norm.pdf(x, 0, 1)


def mvnpdf(x, dim):
    """ Multivariate normal density function used to generate samples using Metropolis-Hastings Algorithm
    :math: `f(x_{1},...,x_{k}) = \\frac{1}{((2*\\pi)^{k}*\\Sigma)^(1/2)}*exp(-\\frac{1}{2}*(x-\\mu)^{T}*\\Sigma^{-1}*(x-\\mu))`

    """
    return stats.multivariate_normal.pdf(x, mean=np.zeros(dim), cov=np.identity(dim))


def marginal(x, mp):
    """
    Marginal target density used to generate samples using Modified Metropolis-Hastings Algorithm

    :math:`f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}}\\exp{-\\frac{1}{2}{\\frac{x-\\mu}{\\sigma}}^2}`

    """
    return stats.norm.pdf(x, mp[0], mp[1])


def srom1(x):
    return stats.gamma.cdf(x, 2, loc=1, scale=3)


def srom2(x):
    return stats.gamma.cdf(x, 2, loc=1, scale=3)


def srom3(x):
    return stats.gamma.cdf(x, 2, loc=1, scale=3)

