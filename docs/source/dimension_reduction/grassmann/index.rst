Grassmann manifold
--------------------------------

In differential geometry the Grassmann manifold :math:`\mathcal{G}(p, n)` refers to the collection of all
:math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space
:cite:`Grassmann_1` :cite:`Grassmann_2`. A point on :math:`\mathcal{G}(p, n)` is typically represented as a
:math:`n \times p` orthonormal matrix :math:`\mathbf{X}`, whose column spans the corresponding subspace. :py:mod:`UQpy`
contains a set of classes and methods for data projection onto the Grassmann manifold, operations and interpolation on
the Grassmann manifold.

.. toctree::
   :maxdepth: 1
   :caption: Methods

    Grassmann Projections <manifold_projections>
    Grassmann Operations <grassmann_operations>
    Grassmann Interpolation <grassmann_interpolation>
    Grassmann Examples <../../auto_examples/dimension_reduction/grassmann/index>
