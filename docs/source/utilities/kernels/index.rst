Kernels
-----------------------------------

A collection of kernel functions in the Euclidean space and the Grassmann manifold that are compatible with a valid metric.


In several applications the use of subspaces is essential to describe the underlying geometry of data. However, it is
well-known that subspaces do not follow the Euclidean geometry because they lie on the Grassmann manifold.Therefore,
working with subspaces requires the definition of an embedding structure of the Grassmann manifold into a Hilbert
space. Thus, using positive definite kernels is studied as a solution to this problem. In this regard, a real-valued
positive definite kernel is defined as a symmetric function :math:`k:\mathcal{X}\times \mathcal{X} \rightarrow \mathbb{R}`
if and only if :math:`\sum^n_{I,j=1}c_i c_j k(x_i,x_j) \leq 0` for :math:`n \in \mathbb{N}`, :math:`x_i \in \mathcal{X}`
and :math:`c_i \in \mathbb{R}`. On the Grassmann manifold a kernel is defined as a well-defined and positive definite
function :math:`k:\mathcal{G}(p,n)\times \mathcal{G}(p,n) \rightarrow \mathbb{R}` :cite:`kernels_1`, :cite:`kernels_2`. A Grassmann kernel is a
well-defined positive definite function that embeds the Grassmannian into a Hilbert space. :py:mod:`UQpy` introduces
two Grassmann kernels have been proposed in literature and have demonstrated the potential for subspace-based learning
problems.


Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.kernels.baseclass.Kernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.Kernel import Kernel

.. autoclass:: UQpy.utilities.kernels.baseclass.Kernel
    :members: kernel_entry, optimize_parameters


.. toctree::
   :maxdepth: 1

    Grassmannian Kernels <grassmann_kernels>
    Euclidean Kernels <euclidean_kernels>
