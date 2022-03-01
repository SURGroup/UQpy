Utilities
================

.. toctree::
   :maxdepth: 1
   :caption: Auxiliary functionalities

    Distances <distances/index>
    Kernels <kernels/index>

Grassmann Point
^^^^^^^^^^^^^^^

The  :py:mod:`UQpy` class :class:`.GrassmannPoint` offers a way to check that a data point, given as a matrix :math:`\mathbf{X} \in \mathbb{R}^{n \times p}`,  belongs on the corresponding Grassmann manifold. To this end, the user needs to create an object of type :class:`.GrassmannPoint`
that will check that the point is given as an orthonormal 2-d numpy.array, i.e., :math:`\text{shape}(\mathbf{X})=(p, n)` and :math:`\mathbf{X}' \mathbf{X} = \mathbf{I}`.
In order to use the class :class:`.GrassmannPoint` one needs to import it

>>> from UQpy.utilities.GrassmannPoint import GrassmannPoint

To create an object of type :class:`.GrassmannPoint`

>>> X = GrassmannPoint(X)

.. autoclass:: UQpy.utilities.GrassmannPoint
