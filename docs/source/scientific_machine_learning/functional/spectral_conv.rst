List of Spectral Convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All spectral convolutions perform the same computation for a signal of different dimensions.

Formula
^^^^^^^

Using the notation from Li 2021, the spectral convolution is defined by

.. math:: SC(x) = \mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )

Note that these functions do not construct :math:`R`,
allowing them to be used in both the deterministic and Bayesian cases.

Spectral Conv 1d
^^^^^^^^^^^^^^^^

The function :func:`spectral_conv1d` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import spectral_conv1d

.. autofunction:: UQpy.scientific_machine_learning.functional.spectral_conv1d

-----

Spectral Conv 2d
^^^^^^^^^^^^^^^^

The function :func:`spectral_conv2d` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import spectral_conv2d

.. autofunction:: UQpy.scientific_machine_learning.functional.spectral_conv2d

-----

Spectral Conv 3d
^^^^^^^^^^^^^^^^

The function :func:`spectral_conv3d` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import spectral_conv3d

.. autofunction:: UQpy.scientific_machine_learning.functional.spectral_conv3d

