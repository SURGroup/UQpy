List of Fourier Layers
======================

All Fourier layers are types of convolutions, although they do not have a direct counterpart in :py:mod:`torch`.

Formula
^^^^^^^

Using the notation from Li 2021, the Fourier layer is defined as

.. math:: FL(x) = \underbrace{\mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )}_\text{Spectral Convolution} + \underbrace{W(x)}_\text{Convolution}

Where the spectral convolution :math:`\mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )` is computed by UQpy's
``sml.functional.spectral_conv`` function and :math:`W` is computed by ``torch.nn.functional.conv``.
The ``Fourier1d`` layer calls ``spectral_conv1d`` and ``conv1d`` and the higher dimensional Fourier layers call the appropriate analogues.

-----

Fourier1d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier1d
    :members: forward

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier1d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier1d.weight_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier1d.weight_conv

-----

Fourier2d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier2d
    :members: forward

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier2d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier2d.weight1_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier2d.weight2_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier2d.weight_conv

-----

Fourier3d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier3d
    :members: forward

.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.weight1_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.weight2_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.weight3_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.weight4_spectral_conv
.. autoattribute:: UQpy.scientific_machine_learning.layers.Fourier3d.weight_conv