Kernels
-----------------------------------

A collection of symmetric positive-definite kernel functions in the Euclidean space and on the Grassmann manifold.

Each kernel function in :py:mod:`UQpy` is defined as a subclass of the :class:`.Kernel` class. The :class:`.Kernel` has
two further subclasses for Euclidean kernels (:class:`.EuclideanKernel`) and Grassmannian kernels
(:class:`.GrassmannianKernel`). Individual kernels, depending on their type, are defined as subclasses of these.

Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.kernels.baseclass.Kernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.Kernel import Kernel

.. autoclass:: UQpy.utilities.kernels.baseclass.Kernel
    :members: kernel_entry, optimize_parameters, calculate_kernel_matrix

Types of Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Kernel` class has subclasses for the following types of kernels:

.. toctree::
   :maxdepth: 1

    Euclidean Kernels <euclidean_kernels>
    Grassmannian Kernels <grassmann_kernels>
