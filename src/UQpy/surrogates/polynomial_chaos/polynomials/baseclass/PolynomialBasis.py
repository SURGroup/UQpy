from abc import ABC
from typing import Union

from UQpy.distributions.baseclass import Distribution
from UQpy.distributions.collection import Uniform, Normal
from UQpy.distributions.collection import JointIndependent, JointCopula
from UQpy.surrogates.polynomial_chaos.polynomials.PolynomialsND import PolynomialsND
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials
from UQpy.utilities import NoPublicConstructor
import itertools
import math
import numpy as np
from scipy.special import comb


class PolynomialBasis(ABC):

    def __init__(self, inputs_number: int,
                 polynomials_number: int,
                 multi_index_set: np.ndarray,
                 polynomials: Polynomials,
                 distributions: Union[Distribution, list[Distribution]]):
        """
        Create polynomial basis for a given multi index set.
        """
        self.polynomials = polynomials
        self.multi_index_set = multi_index_set
        self.polynomials_number = polynomials_number
        self.inputs_number = inputs_number
        self.distributions = distributions

    def evaluate_basis(self, samples: np.ndarray):
        samples_number = len(samples)
        eval_matrix = np.empty([samples_number, self.polynomials_number])
        for ii in range(self.polynomials_number):
            eval_matrix[:, ii] = self.polynomials[ii].evaluate(samples)

        return eval_matrix

    @staticmethod
    def calculate_total_degree_set(inputs_number: int, degree: int):
        # size of the total degree multiindex set
        td_size = int(comb(inputs_number + degree, inputs_number))

        # initialize total degree multiindex set
        midx_set = np.empty([td_size, inputs_number])

        # starting row
        row_start = 0

        # iterate by polynomial order
        for i in range(degree + 1):
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
    def calculate_hyperbolic_set(inputs_number, degree,q):

        xmono=np.zeros(inputs_number)
        X=[]
        X.append(xmono)
        
        while np.sum(xmono)<=degree:
            # generate multi-indices one by one
            x=np.array(xmono)
            i = 0
            for j in range ( inputs_number, 0, -1 ):
                if ( 0 < x[j-1] ):
                    i = j
                    break

            if ( i == 0 ):
                x[inputs_number-1] = 1
                xmono=x
            else:
                if ( i == 1 ):
                    t = x[0] + 1
                    im1 = inputs_number
                if ( 1 < i ):
                    t = x[i-1]
                    im1 = i - 1

                x[i-1] = 0
                x[im1-1] = x[im1-1] + 1
                x[inputs_number-1] = x[inputs_number-1] + t - 1

                xmono=x
                
            # check the hyperbolic criterion          
            if (np.round(np.sum(xmono**q)**(1/q), 4) <= degree):
                X.append(xmono)


        return(np.array(X).astype(int))
    
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
        poly_basis = []
        if inputs_number == 1:
            return [
                Polynomials.distribution_to_polynomial[type(distributions)](
                    distributions=distributions, degree=int(idx[0])) for idx in multi_index_set]
        else:
            return [PolynomialsND(distributions, idx) for idx in multi_index_set]
