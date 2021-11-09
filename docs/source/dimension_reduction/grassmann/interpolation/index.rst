
Piecewise Linear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`.LinearInterpolation` class can be used to  perform piecewise linear interpolation between points according to:

.. math:: p(x) = y_k + \frac{y_{k+1}-y_k}{t_{k+1}-t_k}(x-t_k), \text{for} x \in [t_k, t_{k+1}]

where in each interval :math:`[t_k, t_{k+1}]`, :math:`p(x)` is a linear function.

This class is a child class of the :class:`.InterpolationMethod` class and utilizes the :class:`scipy.interpolate.LinearNDInterpolator` method of
the scipy package to perform interpolation in high dimensions.
To use this class one needs to import it as

>>> from UQpy.dimension_reduction.LinearInterpolation import LinearInterpolation

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.LinearInterpolation
    :members:


Surrogate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This class is a child class of the :class:`.InterpolationMethod` class and utilizes the :class:`.Surrogates` module
to perform interpolation
To use this class one needs to import it as

>>> from UQpy.dimension_reduction.SurrogateInterpolation import SurrogateInterpolation

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.SurrogateInterpolation
    :members:


-----------------------------------------------------------------------------

The abstract :class:`.InterpolationMethod` class is the parent class that allows the user to define an interpolation
method by writing a child classes built from this abstract class.

