List of Normalizer Layers
=========================

All normalizer layers scale and shift tensors, but do *not* have learnable parameters.

GaussianNormalizer
~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.GaussianNormalizer
    :members: forward, encode, decode

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.GaussianNormalizer.encoding
.. autoattribute:: UQpy.scientific_machine_learning.GaussianNormalizer.mean
.. autoattribute:: UQpy.scientific_machine_learning.GaussianNormalizer.std

-----

RangeNormalizer
~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.RangeNormalizer
    :members: forward, encode, decode

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.RangeNormalizer.encoding
.. autoattribute:: UQpy.scientific_machine_learning.RangeNormalizer.scale
.. autoattribute:: UQpy.scientific_machine_learning.RangeNormalizer.shift
