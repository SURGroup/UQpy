from typing import Union

from UQpy.distributions.baseclass import Distribution
from UQpy.distributions.collection import JointIndependent, JointCopula
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis


class TensorProductBasis(PolynomialBasis):

    def __init__(self, distributions: Union[Distribution, list[Distribution]], max_degree: int):
        """
        Create tensor-product polynomial basis. 
        The size is equal to :code:`(max_degree+1)**n_inputs` (exponential complexity).

        :param distributions: List of univariate distributions.
        :param max_degree: Maximum polynomial degree of the 1D chaos polynomials.
        """
        inputs_number = 1 if not isinstance(distributions, (JointIndependent, JointCopula)) \
            else len(distributions.marginals)
        multi_index_set = PolynomialBasis.calculate_tensor_product_set(inputs_number=inputs_number,
                                                                       degree=max_degree)
        polynomials = PolynomialBasis.construct_arbitrary_basis(inputs_number, distributions, multi_index_set)
        super().__init__(inputs_number, len(multi_index_set), multi_index_set, polynomials, distributions)


