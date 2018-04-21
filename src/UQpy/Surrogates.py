import chaospy as cp
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


class SurrogateModels:
    """
    A class containing various surrogate models

    """

########################################################################################################################
########################################################################################################################
#                                        Polynomial Chaos
########################################################################################################################
    class PolynomialChaos:

        """
        A class used to generate a Polynomial Chaos surrogate.

        :param dimension: Dimension of the input space
        :param input:     Input data of shape:  (N x Dimension)
        :param output:    Output data of shape:  (N,)
        :param order:     Order of the polynomial chaos model


        Created by: Dimitris G. Giovanis
        Last modified: 12/10/2017
        Last modified by: Dimitris G. Giovanis

        """

        def __init__(self, dimension=None, input=None, output=None, order=None):

            self.dimension = dimension
            self.input = np.transpose(input)
            self.output = output
            self.order = order

            self.distribution = cp.Iid(cp.Uniform(0, 1), self.dimension)
            orthogonal_expansion = cp.orth_ttr(self.order, self.distribution)
            self.poly = cp.fit_regression(orthogonal_expansion, self.input, self.output)

        def PCpredictor(self, sample):

            if len(sample.shape) == 1:
                g_tilde = 0.0
                if self.dimension == 1:
                    g_tilde = self.poly(sample[0])

                elif self.dimension == 2:
                    g_tilde = self.poly(sample[0], sample[1])

            else:
                g_tilde = np.zeros(sample.shape[0])
                for i in range(sample.shape[0]):
                    if self.dimension == 1:
                        g_tilde[i] = self.poly(sample[i])

                    elif self.dimension == 2:
                        g_tilde[i] = self.poly(sample[i][0], sample[i][1])
                        print()

            return g_tilde


########################################################################################################################
########################################################################################################################
#                                        Gaussian Process
########################################################################################################################
    class GaussianProcess:

        """
        A class used to generate a Gaussian process surrogate.

        :param input:  Input data of shape:  (N x Dimension)
        :param output: Output data of shape:  (N,)


        Created by: Dimitris G. Giovanis
        Last modified: 12/10/2017
        Last modified by: Dimitris G. Giovanis

        """

        def __init__(self, input=None, output=None):

            self.input = input
            self.output = output

            kernel = C(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-3, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
            gp.fit(self.input, self.output)
            self.gp = gp

        def GPredictor(self, sample):

            if len(sample.shape) == 1:
                sample = sample.reshape(-1, 1)
                g_tilde, g_std = self.gp.predict(sample.T, return_std=True)
                return g_tilde[0], g_std[0]
            else:
                g_tilde, g_std = self.gp.predict(sample, return_std=True)
                return g_tilde, g_std
