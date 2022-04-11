Grassmann Point
-----------------------------------

The  :py:mod:`UQpy` class :class:`.GrassmannPoint` offers a way to check whether a data point, given as a matrix
:math:`\mathbf{X} \in \mathbb{R}^{n \times p}`,  belongs on the corresponding Grassmann manifold. The class takes, as
input, an orthonormal 2d :class:`.numpy.ndarray` i.e., :math:`\text{shape}(\mathbf{X})=(p, n)`, and checks if this matrix
is an orthonormal basis that lies with :math:`\mathbf{X}' \mathbf{X} = \mathbf{I}` on the Grassmann manifold. If it is,
then it creates the corresponding :class:`.GrassmannPoint` object.

To use the :class:`.GrassmannPoint` class one needs to first import it by

>>> from UQpy.utilities.GrassmannPoint import GrassmannPoint

To create an object of type :class:`.GrassmannPoint`

>>> X = GrassmannPoint(X)

.. autoclass:: UQpy.utilities.GrassmannPoint
