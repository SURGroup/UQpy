from UQpy.distributions.collection import Uniform, Normal
from UQpy.distributions.collection import JointIndependent, JointCopula
from UQpy.surrogates.polynomial_chaos.polynomials import Hermite, Legendre
from UQpy.surrogates.polynomial_chaos.polynomials.PolynomialsND import PolynomialsND
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials
from UQpy.utilities import NoPublicConstructor
import itertools
import math
import numpy as np
from scipy.special import comb


class PolynomialBasis(metaclass=NoPublicConstructor):

    def __init__(self, inputs_number: int,
                 polynomials_number,
                 multi_index_set,
                 polynomials):
        """
        Create polynomial basis for a given multi index set.
        """
        self.polynomials = polynomials
        self.multi_index_set = multi_index_set
        self.polynomials_number = polynomials_number
        self.inputs_number = inputs_number

    @classmethod
    def create_total_degree_basis(cls, distributions, max_degree):
        """
        Create tensor-product polynomial basis.
        The size is equal to :code:`(max_degree+1)**n_inputs` (exponential complexity).

        :param distributions: List of univariate distributions.
        :param max_degree: Maximum polynomial degree of the 1D chaos polynomials.
        """
        inputs_number = 1 if not isinstance(distributions, (JointIndependent, JointCopula)) \
            else len(distributions.marginals)
        multi_index_set = PolynomialBasis.calculate_total_degree_set(inputs_number=inputs_number,
                                                                     degree=max_degree)
        polynomials = PolynomialBasis.construct_arbitrary_basis(inputs_number, distributions, multi_index_set)
        return cls._create(inputs_number, len(multi_index_set), multi_index_set, polynomials)

    @classmethod
    def create_tensor_product_basis(cls, distributions, max_degree):
        """
        Create total-degree polynomial basis.
        The size is equal to :code:`(total_degree+n_inputs)!/(total_degree!*n_inputs!)`
        (polynomial complexity).

        :param distributions: List of univariate distributions.
        :param max_degree: Maximum polynomial degree of the 1D chaos polynomials.
        """
        inputs_number = 1 if not isinstance(distributions, (JointIndependent, JointCopula)) \
            else len(distributions.marginals)
        multi_index_set = PolynomialBasis.calculate_tensor_product_set(inputs_number=inputs_number,
                                                                       degree=max_degree)
        polynomials = PolynomialBasis.construct_arbitrary_basis(inputs_number, distributions, multi_index_set)
        return cls._create(inputs_number, len(multi_index_set), multi_index_set, polynomials)

    def evaluate_basis(self, samples):
        samples_number = len(samples)
        eval_matrix = np.empty([samples_number, self.polynomials_number])
        for ii in range(self.polynomials_number):
            eval_matrix[:, ii] = self.polynomials[ii].evaluate(samples)

        return eval_matrix

    @staticmethod
    def calculate_total_degree_set(inputs_number, degree):
        # size of the total degree multiindex set
        td_size = int(comb(inputs_number + degree, inputs_number))

        # initialize total degree multiindex set
        midx_set = np.empty([td_size, inputs_number])

        # starting row
        row_start = 0

        # iterate by polynomial order
        for i in range(0, degree + 1):
            # compute number of rows
            rows = PolynomialBasis._setsize(inputs_number, i)

            # update up to row r2
            row_end = rows + row_start - 1

            # recursive call
            midx_set[row_start:row_end + 1, :] = PolynomialBasis. \
                calculate_total_degree_recursive(inputs_number, i, rows)

            # update starting row
            row_start = row_end + 1

        return midx_set.astype(int)

    @staticmethod
    def _setsize(inputs_number, degree):
        return int(comb(inputs_number + degree - 1, inputs_number - 1))

    @staticmethod
    def calculate_total_degree_recursive(N, w, rows):
        if N == 1:
            subset = w * np.ones([rows, 1])
        else:
            if w == 0:
                subset = np.zeros([rows, N])
            elif w == 1:
                subset = np.eye(N)
            else:
                # initialize submatrix
                subset = np.empty([rows, N])

                # starting row of submatrix
                row_start = 0

                # iterate by polynomial order and fill the multiindex submatrices
                for k in range(0, w + 1):
                    # number of rows of the submatrix
                    sub_rows = PolynomialBasis._setsize(N - 1, w - k)

                    # update until row r2
                    row_end = row_start + sub_rows - 1

                    # first column
                    subset[row_start:row_end + 1, 0] = k * np.ones(sub_rows)

                    # subset update --> recursive call
                    subset[row_start:row_end + 1, 1:] = \
                        PolynomialBasis.calculate_total_degree_recursive(N - 1, w - k, sub_rows)

                    # update row indices
                    row_start = row_end + 1

        return subset

    @staticmethod
    def calculate_tensor_product_set(inputs_number, degree):
        orders = np.arange(0, degree + 1, 1).tolist()
        if inputs_number == 1:
            midx_set = np.array(list(map(lambda el: [el], orders)))
        else:
            midx = list(itertools.product(orders, repeat=inputs_number))
            midx = [list(elem) for elem in midx]
            midx_sums = [int(math.fsum(midx[i])) for i in range(len(midx))]
            midx_sorted = sorted(range(len(midx_sums)),
                                 key=lambda k: midx_sums[k])
            midx_set = np.array([midx[midx_sorted[i]] for i in range(len(midx))])
        return midx_set.astype(int)

    @staticmethod
    def construct_arbitrary_basis(inputs_number, distributions, multi_index_set):
        # populate polynomial basis
        poly_basis = list()
        if inputs_number == 1:
            poly_basis = [PolynomialBasis.distribution_to_polynomial[type(distributions)]
                          (distributions=distributions, degree=int(idx[0])) for idx in multi_index_set]
        else:
            poly_basis = [PolynomialsND(distributions, idx) for idx in multi_index_set]
        return poly_basis

    distribution_to_polynomial = {
        Uniform: Legendre,
        Normal: Hermite
    }