Simplex
-------

The :class:`.Simplex` class generates uniformly distributed samples inside a simplex of dimension :math:`n_d`, whose coordinates are expressed by :math:`\zeta_k`. First, this class generates :math:`n_d` independent uniform random variables on [0, 1], denoted :math:`r_q`, then maps them to the simplex as follows:

.. math:: \mathbf{M_{n_d}} = \zeta_0 + \sum_{i=1}^{n_d} \Big{[}\prod_{j=1}^{i} r_{n_d-j+1}^{\frac{1}{n_d-j+1}}\Big{]}(\zeta_i - \zeta_{i-1})

where :math:`M_{n_d}` is an :math:`n_d` dimensional array defining the coordinates of new sample. This mapping is illustrated below for a two-dimensional simplex.

.. image:: ../_static/SampleMethods_Simplex.png
   :scale: 50 %
   :alt: Randomly generated point inside a 2-D simplex
   :align: center

Additional details can be found in :cite:`Simplex1`.

Simplex Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Methods
~~~~~~~~~~~~~~~~~~
.. autoclass:: UQpy.sampling.SimplexSampling
    :members: run

Attributes
~~~~~~~~~~~~~~~~~~
.. autoattribute:: UQpy.sampling.SimplexSampling.samples

.. toctree::
   :hidden:

   Examples <../auto_examples/sampling/simplex/index>