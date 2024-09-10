List of Fourier Layers
======================

All Fourier layers are types of convolutions, although they do not have a direct counterpart in :py:mod:`torch`.

Formula
^^^^^^^

Using the notation from Li 2021, the Fourier layer is defined as

.. math:: FL(x) = \underbrace{\mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )}_\text{Spectral Convolution} + \underbrace{W(x)}_\text{Convolution}


The spectral convolution :math:`\mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )` is computed by UQpy's
``sml.functional.spectral_conv`` function and :math:`W(x)` is computed by ``torch.nn.functional.conv``.
The ``Fourier1d`` layer calls ``spectral_conv1d`` and ``conv1d`` and the higher dimensional Fourier layers call the
appropriate higher-dimensional functions.

The forward Fourier transform :math:`\mathcal{F}` and its inverse :math:`\mathcal{F}^{-1}` are computed by
:py:class:`torch.fft`.
The linear transformation :math:`R` is the learnable parameter :py:attr:`weight_spectral`.
:py:attr:`weight_spectral` contains real numbers (:py:class:`torch.float`) that are cast to complex
(:py:class:`torch.cfloat`) with 0 in the imaginary component for compatibility with the spectral convolutions computed
by the Scientific Machine Learning :code:`functional` submodule.
The convolution :math:`W` is computed by the appropriate convolution from :py:class:`torch.nn.functional` using the
learnable weights :py:attr:`weight_conv` and optional bias :py:attr:`bias_conv`.

-----

Fourier1d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier1d
    :members: forward

-----

Fourier2d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier2d
    :members: forward

-----

Fourier3d
~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.Fourier3d
    :members: forward
