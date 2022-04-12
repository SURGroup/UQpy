Polynomials
""""""""""""""""""""""""""""

Different families of univariate polynomials can be used for the PCE method. These polynomials must always be orthonormal
with respect to the arbitrary distribution. In UQpy, two families of polynomials are currently available that can be
used from their corresponding classes, namely the :class:`.Legendre` and :class:`.Hermite` polynomial class, appropriate for
data generated from a Uniform and a Normal distribution respectively.

The :class:`.Polynomials` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials
    :members:

Legendre Class
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.Legendre` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.Legendre import Legendre

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.Legendre
    :members:

Hermite Class
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.Hermite` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.Hermite import Hermite

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.Hermite
    :members:

PolynomialsND Class
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.PolynomialsND` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.polynomials.PolynomialsND import PolynomialsND

.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.PolynomialsND
    :members:
