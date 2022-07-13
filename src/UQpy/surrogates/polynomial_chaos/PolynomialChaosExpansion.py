import logging

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import PolynomialBasis
from UQpy.distributions import Uniform, Normal
from UQpy.surrogates.polynomial_chaos.polynomials import Legendre, Hermite


class PolynomialChaosExpansion(Surrogate):

    @beartype
    def __init__(self, polynomial_basis: PolynomialBasis, regression_method: Regression):
        """
        Constructs a surrogate model based on the Polynomial Chaos Expansion (polynomial_chaos) method.

        :param regression_method: object for the method used for the calculation of the polynomial_chaos coefficients.
        """
        self.polynomial_basis: PolynomialBasis = polynomial_basis
        """Contains the 1D or ND chaos polynomials that form the PCE basis"""
        self.multi_index_set: NumpyFloatArray = polynomial_basis.multi_index_set
        "Multi-index-set"
        self.regression_method = regression_method
        self.logger = logging.getLogger(__name__)
        self.coefficients: NumpyFloatArray = None
        """Polynomial Chaos Expansion Coefficient"""
        self.bias: NumpyFloatArray = None
        """Bias term in case LASSO or ridge regression are employed to estimate 
        the PCE coefficients"""
        self.outputs_number: int = None
        """Dimensions of the quantities of interest"""
        self.design_matrix: NumpyFloatArray = None
        """Matrix containing the evaluations of the PCE basis on the experimental 
        design that has been used to fit the PCE coefficients"""

        self.experimental_design_input: NumpyFloatArray = None
        """Realizations of the random parameter in the experimental design that 
        has been used to fit the PCE coefficients"""
        self.experimental_design_output: NumpyFloatArray = None
        """Model outputs for the random parameter realizations of the 
        experimental design that has been used to fit the PCE coefficients"""

    @property
    def polynomials_number(self):
        return len(self.polynomial_basis.polynomials)

    @property
    def inputs_number(self):
        return self.polynomial_basis.inputs_number

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the surrogate model using the training samples and the corresponding model values. This method calls the
        :py:meth:`run` method of the input method class.

        :param x: containing the training points.
        :param y: containing the model evaluations at the training points.

        The :meth:`fit` method has no returns and it creates an :class:`numpy.ndarray` with the
        polynomial_chaos coefficients.
        """
        self.experimental_design_input = x
        self.experimental_design_output = y
        self.design_matrix = self.polynomial_basis.evaluate_basis(x)
        self.logger.info("UQpy: Running polynomial_chaos.fit")
        self.coefficients, self.bias, self.outputs_number = self.regression_method.run(x, y, self.design_matrix)
        self.logger.info("UQpy: polynomial_chaos fit complete.")

    def predict(self, points: np.ndarray, **kwargs: dict):
        """
        Predict the model response at new points.
        This method evaluates the polynomial_chaos model at new sample points.

        :param points: Points at which to predict the model response.
        :return: Predicted values at the new points.
        """
        a = self.polynomial_basis.evaluate_basis(points)
        y = a.dot(self.coefficients)
        if self.bias is not None:
            y = y + self.bias
        return y

    def leaveoneout_error(self):
        """
        Returns the cross validation error (leave-one-out) based on experimental design.

        The :meth:`.PolynomialChaosExpansion.leaveoneout_error` method can be used to estimate the accuracy of the PCE
        predictor without additional simulations. Leave-one-out error :math:`Q^2` is calculated from differences between the original mathematical model and approximation :math:`\Delta_i= \mathcal{M} (x^{(i)}) - \mathcal{M}^{PCE}(x^{(i)})` as follows:
        
        .. math:: Q^2 = \mathbb{E}[(\Delta_i/(1-h_i))^2]/\sigma_Y^2
        

        where :math:`\sigma_Y^2` is a variance of model response and :math:`h_i` represents the :math:`i` th diagonal term of matrix :math:`\mathbf{H}= \Psi ( \Psi^T \Psi )^{-1} \Psi^T` obtained from design matrix :math:`\Psi` without additional simulations  :cite:`BLATMANLARS`.

        :return: Cross validation error of experimental design.
        """

        x = self.experimental_design_input
        y = self.experimental_design_output

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        y_val = self.predict(x, )
        polynomialbasis = self.design_matrix

        H = np.dot(polynomialbasis, np.linalg.pinv(np.dot(polynomialbasis.T, polynomialbasis)))
        H *= polynomialbasis
        Hdiag = np.sum(H, axis=1).reshape(-1, 1)

        eps_val = ((n_samples - 1) / n_samples * np.sum(((y - y_val) / (1 - Hdiag)) ** 2, axis=0)) / (
            np.sum((y - mu_yval) ** 2, axis=0))
        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)

    def validation_error(self, x: np.ndarray, y: np.ndarray):
        """
        Returns the validation error.

        The :meth:`.PolynomialChaosExpansion.validation_error` method can be used to estimate the accuracy of the PCE
        predictor.
        Here, we compute the generalization error  in the form of the relative mean squared error normalized by the
        model variance.
        The user must create an independent validation dataset :math:`[x_{val}, y_{val} = M(x_{val})]`
        (i.e. a set of inputs and outputs of the computational model).

        In case where the computational model is very expensive, the use of an alternative error measure is recommended,
        for example the cross-validation error which partitions the existing training dataset into subsets and computes
        the error as the average of the individual errors of each subset.

        :param x: :class:`numpy.ndarray` containing the samples of the validation dataset.
        :param y: :class:`numpy.ndarray` containing model evaluations for the validation dataset.
        :return: Validation error.
        """

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        y_val = self.predict(x, )

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = ((n_samples - 1) / n_samples
                   * ((np.sum((y - y_val) ** 2, axis=0))
                      / (np.sum((y - mu_yval) ** 2, axis=0))))

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)

    def get_moments(self, higher: bool = False):
        """
        Returns the first four moments of the polynomial_chaos surrogate which are directly
        estimated from the polynomial_chaos coefficients.

        The :meth:`.PolynomialChaosExpansion.get_moments` method can be used for the calculation of the first two moments of the PCE model directly from the PCE coefficients. This is possible due to the orthonormality of the polynomial basis.

        The first moment (mean value) is calculated as

        .. math:: \mu_{PCE} = \mathbb{E} [ \mathcal{M}^{PCE}(x)] = y_{0}

        where :math:`y_{0}` is the first PCE coefficient associated with the constant term.

        The second moment (variance) is calculated as

        .. math:: \sigma^{2}_{PCE} = \mathbb{E} [( \mathcal{M}^{PCE}(x) - \mu_{PCE} )^{2} ] = \sum_{i=1}^{p} y_{i}

        where :math:`p` is the number of polynomials (first PCE coefficient is excluded).
        
        The third moment (skewness) and the fourth moment (kurtosis) are generally obtained from third and fourth order products obtained by integration, which are extremely computationally demanding. Therefore here we use analytical solution of standard linearization problem for Hermite (based on normalized Feldheim's formula ) and Legendre polynomials (based on normalized Neumann-Adams formula), though it might be still computationally demanding for high number of input random variables.
        
        :param higher: True corresponds to calculation of skewness and kurtosis (computationaly expensive for large basis set).
        :return: Returns the mean and variance.
        """

        if self.bias is not None:
            mean = self.coefficients[0, :] + np.squeeze(self.bias)
        else:
            mean = self.coefficients[0, :]

        variance = np.sum(self.coefficients[1:] ** 2, axis=0)

        if self.coefficients.ndim == 1 or self.coefficients.shape[1] == 1:
            variance = float(variance)
            mean = float(mean)

        if not higher:
            return mean, variance

        else:
            multindex = self.multi_index_set
            P, inputs_number = multindex.shape

            if inputs_number == 1:
                marginals = [self.polynomial_basis.distributions]

            else:
                marginals = self.polynomial_basis.distributions.marginals

            skewness = np.zeros(self.outputs_number)
            kurtosis = np.zeros(self.outputs_number)

            for ii in range(0, self.outputs_number):

                Beta = self.coefficients[:, ii]
                third_moment = 0
                fourth_moment = 0

                indices = np.array(np.meshgrid(range(1, P), range(1, P), range(1, P), range(1, P))).T.reshape(-1, 4)
                i = 0
                for index in indices:
                    tripleproduct_ND = 1
                    quadproduct_ND = 1

                    for m in range(0, inputs_number):

                        if i < (P - 1) ** 3:

                            if type(marginals[m]) == Normal:
                                tripleproduct_1D = Hermite.hermite_triple_product(multindex[index[0], m],
                                                                                  multindex[index[1], m],
                                                                                  multindex[index[2], m])

                            if type(marginals[m]) == Uniform:
                                tripleproduct_1D = Legendre.legendre_triple_product(multindex[index[0], m],
                                                                                    multindex[index[1], m],
                                                                                    multindex[index[2], m])

                            tripleproduct_ND = tripleproduct_ND * tripleproduct_1D

                        else:
                            tripleproduct_ND = 0

                        quadproduct_1D = 0

                        for n in range(0, multindex[index[0], m] + multindex[index[1], m] + 1):

                            if type(marginals[m]) == Normal:
                                tripproduct1 = Hermite.hermite_triple_product(multindex[index[0], m],
                                                                              multindex[index[1], m], n)
                                tripproduct2 = Hermite.hermite_triple_product(multindex[index[2], m],
                                                                              multindex[index[3], m], n)

                            if type(marginals[m]) == Uniform:
                                tripproduct1 = Legendre.legendre_triple_product(multindex[index[0], m],
                                                                                multindex[index[1], m], n)
                                tripproduct2 = Legendre.legendre_triple_product(multindex[index[2], m],
                                                                                multindex[index[3], m], n)

                            quadproduct_1D = quadproduct_1D + tripproduct1 * tripproduct2

                        quadproduct_ND = quadproduct_ND * quadproduct_1D

                    third_moment += tripleproduct_ND * Beta[index[0]] * Beta[index[1]] * Beta[index[2]]
                    fourth_moment += quadproduct_ND * Beta[index[0]] * Beta[index[1]] * Beta[index[2]] * Beta[index[3]]

                    i += 1

                skewness[ii] = 1 / (np.sqrt(variance) ** 3) * third_moment
                kurtosis[ii] = 1 / (variance ** 2) * fourth_moment

                if self.coefficients.ndim == 1 or self.coefficients.shape[1] == 1:
                    skewness = float(skewness[0])
                    kurtosis = float(kurtosis[0])

            return mean, variance, skewness, kurtosis
