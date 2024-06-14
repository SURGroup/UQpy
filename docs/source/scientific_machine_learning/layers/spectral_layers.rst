List of Spectral Convolution Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a placeholder for the documentation on Spectral Convolution layers.

Formula
^^^^^^^

Using the notation from Li 2021, the spectral convolution is defined by

.. math:: SC(x) = \mathcal{F}^{-1}( R ( \mathcal{F}(x) ) )

-----

Spectral Convolution 1D
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.scientific_machine_learning.layers.SpectralConv1d
    :members: forward

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv1d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv1d.weights

-----

Spectral Convolution 2D
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.scientific_machine_learning.layers.SpectralConv2d
    :members: forward

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv2d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv2d.weights1
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv2d.weights2

-----

Spectral Convolution 3D
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.scientific_machine_learning.layers.SpectralConv3d
    :members: forward

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv3d.scale
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv3d.weights1
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv3d.weights2
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv3d.weights3
.. autoattribute:: UQpy.scientific_machine_learning.layers.SpectralConv3d.weights4
