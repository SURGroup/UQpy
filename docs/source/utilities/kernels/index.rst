Kernels
-----------------------------------

A collection of symmetric positive-definite kernel functions in the Euclidean space and on the Grassmann manifold.

A real-valued positive definite kernel is defined as a symmetric function
:math:`k:\mathcal{X}\times \mathcal{X} \rightarrow \mathbb{R}`
where :math:`\sum^n_{i,j=1}c_i c_j k(x_i,x_j) \leq 0` for :math:`n \in \mathbb{N}`, :math:`x_i \in \mathcal{X}`
and :math:`c_i \in \mathbb{R}`.

Each kernel function in :py:mod:`UQpy` is defined as a subclass of the :class:`UQpy.utilities.kernels.baseclass.Kernel`
class. The :class:`UQpy.utilities.kernels.baseclass.Kernel` has two further subclasses for Euclidean kernels (:class:`.EuclideanKernel`) and Grassmannian kernels
(:class:`.GrassmannianKernel`). Individual kernels, depending on their type, are defined as subclasses of these.

Kernel Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.kernels.baseclass.Kernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.Kernel import Kernel

.. autoclass:: UQpy.utilities.kernels.baseclass.Kernel
    :members: calculate_kernel_matrix

Types of Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.kernels.baseclass.Kernel` class has subclasses for the following types of kernels:

.. toctree::
   :maxdepth: 1

    Euclidean Kernels <euclidean_kernels>
    Grassmannian Kernels <grassmann_kernels>
    Sum and Product Kernels <sum_product_kernels>
