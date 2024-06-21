List of Fourier Layers
======================

All Fourier layers are types of convolutions, although they do not have a direct counterpart in :py:mod:`torch`.

Formula
^^^^^^^

Using the notation from Li 2021, the spectral convolution is defined by

.. math:: SC(x) = \mathcal{F}^{-1}( R ( \mathcal{F}(x) ) ) + W

Where the spectral convolution :math:`\mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )` is computed by UQpy's
``SpectralConv`` class and :math:`W` is computed by ``torch.nn.Conv``, each of the appropriate dimension.
Note that these functions do not construct :math:`R` or :math:`W`,
allowing them to be used in both the deterministic and Bayesian cases.


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