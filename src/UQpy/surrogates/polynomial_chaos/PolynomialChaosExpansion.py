import logging

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression
from UQpy.surrogates.polynomial_chaos.polynomials.PolynomialBasis import PolynomialBasis
from UQpy.distributions import Uniform, Normal
from UQpy.surrogates.polynomial_chaos.polynomials import Legendre,Hermite

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

    def fit(self, x, y):
        """
        Fit the surrogate model using the training samples and the corresponding model values. This method calls the
        :py:meth:'run' method of the input method class.

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

    def predict(self, points, **kwargs):
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

        :return: Cross validation error of experimental design.
        """
        x=self.experimental_design_input
        y=self.experimental_design_output
        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        y_val = self.predict(x, )
        polynomialbasis= self.design_matrix
        
        H = np.dot(polynomialbasis, np.linalg.inv(np.dot(polynomialbasis.T, polynomialbasis)))
        H *= polynomialbasis
        Hdiag = np.sum(H, axis=1)
        
        eps_val=((n_samples - 1) / n_samples * (np.sum(((y - y_val)/(1 - Hdiag))**2) / n_samples) /  (np.sum((y - mu_yval) ** 2, axis=0)))
        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)
    
    def validation_error(self, x, y):
        """
        Returns the validation error.

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

    def get_moments(self, higher=False):
        """
        Returns the first four moments of the polynomial_chaos surrogate which are directly
        estimated from the polynomial_chaos coefficients.
        
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

        if higher==False:
            return np.round(mean, 4), np.round(variance, 4)
        else:
            multindex=self.multi_index_set
            P,inputs_number=multindex.shape
            if inputs_number==1:
                marginals=[self.polynomial_basis.distributions]
            else:
                marginals=self.polynomial_basis.distributions.marginals
            

            skewness=np.zeros(self.outputs_number)
            kurtosis=np.zeros(self.outputs_number)
            for ii in range (0,self.outputs_number):
                Beta=self.coefficients[:, ii]
                TripleProduct=np.ones((P,P,P))
                ThirdMoment=np.ones((P,P,P))
                QuadIval=np.ones((P,P,P,P))
                Quadproduct=np.zeros((P,P,P,P))
                skewsum=0
                kurtosissum=0
                for p1 in range (1,P):
                    for p2 in range (1,P):
                        for p3 in range (1,P):
                            for m in range (0,inputs_number):
                                if type(marginals[m])==Normal:
                                    tripleproduct=Hermite.hermite_triple_product(multindex[p1,m],multindex[p2,m],multindex[p3,m])
                                if type(marginals[m])==Uniform:   
                                    tripleproduct=Legendre.legendre_triple_product(multindex[p1,m],multindex[p2,m],multindex[p3,m])
                                TripleProduct[p1,p2,p3]=TripleProduct[p1,p2,p3]*tripleproduct

                            ThirdMoment[p1,p2,p3]=TripleProduct[p1,p2,p3]*Beta[p1]*Beta[p2]*Beta[p3]
                            skewsum+=ThirdMoment[p1,p2,p3]

                            for p4 in range(1,P):
                                for m in range (0,inputs_number):
                                    Quadproduct[p1,p2,p3,p4]=0

                                    for n in range (0,multindex[p1,m]+multindex[p2,m]+1):
                                        if type(marginals[m])==Normal:
                                            tripproduct1=Hermite.hermite_triple_product(multindex[p1,m],multindex[p2,m],n)
                                            tripproduct2=Hermite.hermite_triple_product(multindex[p3,m],multindex[p4,m],n)
                                        if type(marginals[m])==Uniform: 
                                            tripproduct1=Legendre.legendre_triple_product(multindex[p1,m],multindex[p2,m],n)
                                            tripproduct2=Legendre.legendre_triple_product(multindex[p3,m],multindex[p4,m],n)

                                        Quadproduct[p1,p2,p3,p4]=Quadproduct[p1,p2,p3,p4]+tripproduct1*tripproduct2

                                    QuadIval[p1,p2,p3,p4]=QuadIval[p1,p2,p3,p4]*Quadproduct[p1,p2,p3,p4]

                                kurtosissum+=QuadIval[p1,p2,p3,p4]*Beta[p1]*Beta[p2]*Beta[p3]*Beta[p4]

                skewness[ii]=1/(np.sqrt(variance)**3)*skewsum
                kurtosis[ii]=1/(variance**2)*kurtosissum
                if self.coefficients.ndim == 1 or self.coefficients.shape[1] == 1:
                    skewness = float(skewness[0])
                    kurtosis = float(kurtosis[0])
            
            
            return mean,variance,skewness,kurtosis
