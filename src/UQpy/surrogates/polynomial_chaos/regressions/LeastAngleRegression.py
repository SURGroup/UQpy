import logging
import numpy as np
from beartype import beartype
import copy

from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import PolynomialBasis
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression
from UQpy.surrogates.polynomial_chaos.regressions import LeastSquareRegression

from sklearn import linear_model as regresion


class LeastAngleRegression(Regression):
    @beartype
    def __init__(self, fit_intercept: bool = False, verbose: bool = False, n_nonzero_coefs: int = 1000,
                 normalize: bool = False):
        """
        Class to select the best model approximation and calculate the polynomial_chaos coefficients with the Least Angle 
        Regression method combined with ordinary least squares.

        :param n_nonzero_coefs: Maximum number of non-zero coefficients.
        :param fit_intercept: Whether to calculate the intercept for this model. Recommended false for PCE, since
         intercept is included in basis functions.
        :param verbose: Sets the verbosity amount.
        """
        self.fit_intercept = fit_intercept
        self.n_nonzero_coefs = n_nonzero_coefs
        self.normalize = normalize
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def run(self, x: np.ndarray, y: np.ndarray, design_matrix: np.ndarray):
        """
        Implements the LAR method to compute the polynomial_chaos coefficients. 
        Recommended only for model_selection algorithm.

        :param x: :class:`numpy.ndarray` containing the training points (samples).
        :param y: :class:`numpy.ndarray` containing the model evaluations (labels) at the training points.
        :param design_matrix: matrix containing the evaluation of the polynomials at the input points **x**.
        :return: Beta (polynomial_chaos coefficients)
        """
        polynomialbasis = design_matrix
        P = polynomialbasis.shape[1]
        n_samples, inputs_number = x.shape

        reg = regresion.Lars(fit_intercept=self.fit_intercept, verbose=self.verbose,
                             n_nonzero_coefs=self.n_nonzero_coefs, normalize=self.normalize)
        reg.fit(design_matrix, y)

        # LarsBeta = reg.coef_path_
        c_ = reg.coef_

        self.Beta_path = reg.coef_path_

        if c_.ndim == 1:
            c_ = c_.reshape(-1, 1)

        return c_, None, np.shape(c_)[1]

    @staticmethod
    def model_selection(pce_object: PolynomialChaosExpansion, target_error=1, check_overfitting=True):
        """
        LARS model selection algorithm for given TargetError of approximation
        measured by Cross validation: Leave-one-out error (1 is perfect approximation). Option to check overfitting by 
        empirical rule: if three steps in a row have a decreasing accuracy, stop the algorithm.

        :param PolynomialChaosExpansion: existing target PCE for model_selection
        :param target_error: Target error of an approximation (stoping criterion).
        :param check_overfitting: Whether to check over-fitting by empirical rule.
        :return: copy of input PolynomialChaosExpansion containing the best possible model for given data identified by LARs  
        """

        pce = copy.deepcopy(pce_object)
        x = pce.experimental_design_input
        y = pce.experimental_design_output

        pce.regression_method = LeastAngleRegression()
        pce.fit(x, y)

        LarsBeta = pce.regression_method.Beta_path
        P, steps = LarsBeta.shape

        polynomialbasis = pce.design_matrix
        multindex = pce.multi_index_set

        pce.regression_method = LeastSquareRegression()

        larsbasis = []
        OLSBetaList = []
        larsindex = []

        LarsError = []
        error = 0
        overfitting = False
        BestLarsError = 0
        step = 0
        
        if steps<3:
            raise Exception('LAR identified constant function! Check your data.')

        while BestLarsError < target_error and step < steps - 2 and overfitting == False:

            mask = LarsBeta[:, step + 2] != 0
            mask[0] = True

            larsindex.append(multindex[mask, :])
            larsbasis.append(list(np.array(pce_object.polynomial_basis.polynomials)[mask]))

            pce.polynomial_basis.polynomials_number = len(larsbasis[step])
            pce.polynomial_basis.polynomials = larsbasis[step]
            pce.multi_index_set = larsindex[step]

            pce.fit(x, y)
            coefficients = pce.coefficients

            LarsError.append(float(1 - pce.leaveoneout_error()))

            error = LarsError[step]

            if step == 0:
                BestLarsMultindex = larsindex[step]
                BestLarsBasis = larsbasis[step]
                BestLarsError = LarsError[step]

            else:
                if error > BestLarsError:
                    BestLarsMultindex = larsindex[step]
                    BestLarsBasis = larsbasis[step]
                    BestLarsError = LarsError[step]

            if (step > 3) and (check_overfitting == True):
                if (BestLarsError > 0.6) and (error < LarsError[step - 1]) and (error < LarsError[step - 2]) and (
                        error < LarsError[step - 3]):
                    overfitting = True

            step += 1

        pce.polynomial_basis.polynomials_number = len(BestLarsBasis)
        pce.polynomial_basis.polynomials = BestLarsBasis
        pce.multi_index_set = BestLarsMultindex

        pce.fit(x, y)

        return pce
