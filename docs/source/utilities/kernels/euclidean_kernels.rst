Euclidean Kernels
-----------------------------------

Euclidean Kernel Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.EuclideanKernel` class is the parent class for all Euclidean kernels. It is imported using the following
command:

>>> from UQpy.utilities.kernels.baseclass.EuclideanKernel import EuclideanKernel

.. autoclass:: UQpy.utilities.kernels.baseclass.EuclideanKernel
    :members: calculate_kernel_matrix

Gaussian Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Gaussian kernel is defined by:

.. math:: k(\mathbf{x}_i,\mathbf{x}_j) = \exp\left(\dfrac{||\mathbf{x}_i-\mathbf{x}_j||^2}{2\epsilon^2}\right)

The :class:`.GaussianKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.GaussianKernel import GaussianKernel

One can use the following to instantiate the :class:`.GaussianKernel` class.

Methods
~~~~~~~~~

.. autoclass:: UQpy.utilities.kernels.GaussianKernel
    :members: kernel_entry, optimize_parameters

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.kernels.GaussianKernel.kernel_matrix