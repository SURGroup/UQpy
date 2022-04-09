Polynomial Bases
""""""""""""""""""""""""""""

The :class:`.PolynomialBasis` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis
    :members:

TotalDegreeBasis Class
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.TotalDegreeBasis` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis
    :members:

TensorProductBasis Class
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.TensorProductBasis` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.TensorProductBasis import TensorProductBasis

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.TensorProductBasis
    :members:

HyperbolicBasis Class
~~~~~~~~~~~~~~~~~~~~~~~~

According to effect-of-sparsity, it is often beneficial to neglect higher-order interaction terms in basis set using hyperbolic truncation :cite:`BLATMANLARS`. 

The selection of a reducing parameter :math:`q=1` corresponds to the total-degree truncation scheme according to and, for :math:`q<1`, terms representing higher-order interactions are eliminated. Such an approach leads to a~dramatic reduction in the cardinality of the truncated set for high total polynomial orders :math:`p` and high input dimensions :math:`M`. Set of basis functions :math:`\mathcal{A}` defined by multi-indices :math:`\alpha` is obtained as:
        
.. math:: \mathcal A^{M,p,q}= \{ \alpha \in \mathbb{N}^{M} : || \alpha ||_q \equiv ( \sum_{i=1}^{M} \alpha_i^q )^{1/q}  \leq p \}.

The :class:`.HyperbolicBasis` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.HyperbolicBasis import HyperbolicBasis

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.HyperbolicBasis
    :members:





