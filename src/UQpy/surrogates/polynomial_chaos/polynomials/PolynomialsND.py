import numpy as np

from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials


class PolynomialsND(Polynomials):

    def __init__(self, distributions, multi_index):
        """
        Class for multivariate Wiener-Askey chaos polynomials.

        :param distributions: Joint probability distribution.
        :param multi_index: Polynomial multi-degree (multi-index).
        """
        self.multi_index = multi_index
        self.distributions = distributions
        marginals = distributions.marginals
        N = len(multi_index)  # dimensions
        self.polynomials1d = [Polynomials.distribution_to_polynomial[type(marginals[n])]
                              (distributions=marginals[n], degree=int(multi_index[n])) for n in range(N)]

    def evaluate(self, eval_data) ->np.ndarray:
        """
        Evaluate Nd chaos polynomial on the given data set.

        :param eval_data: Points upon which the ND chaos polynomial will be evaluated.
        :return: Evaluations of the ND chaos polynomial.
        """
        try:  # case: 2d array, K x N, N being the number of dimensions
            K, N = np.shape(eval_data)
        except:  # case: 1d array, 1 x N, N being the number of dimensions
            K = 1
            N = len(eval_data)
            eval_data = eval_data.reshape(K, N)

        # Store evaluations of 1d polynomials in a KxN matrix. Each column has
        # the evaluations of the n-th 1d polynomial on the n-th data column,
        # i.e. on the values of the n-th parameter
        eval_matrix = np.empty([K, N])
        for n in range(N):
            eval_matrix[:, n] = self.polynomials1d[n].evaluate(eval_data[:, n])

        # The output of the multivariate polynomial is the product of the
        # outputs of the corresponding 1d polynomials
        return np.prod(eval_matrix, axis=1)