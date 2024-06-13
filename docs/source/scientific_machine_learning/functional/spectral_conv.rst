List of Spectral Convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All spectral convolutions perform the same computation for a signal of different dimensions.

Formula
^^^^^^^

Using the notation from Li 2021, the spectral convolution is defined by

.. math:: SC(x) = \mathcal{F}^{-1}( R ( \mathcal{F}(x) ) ) + W

Spectral Conv 1d
^^^^^^^^^^^^^^^^

The  function :func:`spectral_conv1d` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import spectral_conv1d

.. autoclass:: UQpy.scientific_machine_learning.functional.spectral_conv1d

-----

Spectral Conv 2d
^^^^^^^^^^^^^^^^

The function :func:`spectral_conv2d` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import spectral_conv2d

.. py:function:: UQpy.scientific_machine_learning.functional.spectral_conv2d
