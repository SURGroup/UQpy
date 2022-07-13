from typing import Union

import numpy as np

from UQpy.distributions.baseclass import Distribution
from UQpy.distributions.collection import JointIndependent, JointCopula
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis


class HyperbolicBasis(PolynomialBasis):

    def __init__(self, distributions: Union[Distribution, list[Distribution]], max_degree: int, hyperbolic: float = 1):
        """
        Create hyperbolic set from total-degree polynomial basis set.
        
        :param distributions: List of univariate distributions.
        :param max_degree: Maximum polynomial degree of the 1D chaos polynomials.
        :param hyperbolic: Parameter of hyperbolic truncation reducing interaction terms <0,1>
        """
        inputs_number = 1 if not isinstance(distributions, (JointIndependent, JointCopula)) \
            else len(distributions.marginals)
        multi_index_set = PolynomialBasis.calculate_hyperbolic_set(inputs_number=inputs_number,
                                                                     degree=max_degree,q=hyperbolic)
        if 0 < hyperbolic < 1:
            mask = np.round(np.sum(multi_index_set ** hyperbolic, axis=1) ** (1 / hyperbolic), 4) <= max_degree
            multi_index_set = multi_index_set[mask]
        polynomials = PolynomialBasis.construct_arbitrary_basis(inputs_number, distributions, multi_index_set)
        super().__init__(inputs_number, len(multi_index_set), multi_index_set, polynomials, distributions)
