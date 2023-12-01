GP Sobol indices
----------------
The :class:`.GPSobolSensitivity` computes Sobol sensitivity indices using a Gaussian Process regression model as
originally derived in :cite:`MARREL2009742`.

GPSobolSensitivity Class
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GPSobolSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.GPSobolSensitivity import GPSobolSensitivity

Methods
"""""""""""
.. autoclass:: UQpy.sensitivity.GPSobolSensitivity
    :members: run

Attributes
"""""""""""
.. autoattribute:: UQpy.sensitivity.GPSobolSensitivity.sobol_mean
.. autoattribute:: UQpy.sensitivity.GPSobolSensitivity.sobol_std

Examples
""""""""""

.. toctree::

   Gaussian Process Sensitivity Examples <../auto_examples/sensitivity/gpsobol/index>