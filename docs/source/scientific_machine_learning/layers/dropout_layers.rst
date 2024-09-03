List of Dropout Layers
~~~~~~~~~~~~~~~~~~~~~~

All dropout layers are nearly identical implementations to their counterparts in Pytorch.
The difference is these class have a ``dropping`` attribute that controls whether or not they are
active, rather than relying on the ``training`` attribute as Pytorch's implementations do.

This allows us to more conveniently call the dropout methods on forward calls of a neural network,
which is helpful for various computations in uncertainty quantification.

Dropout
^^^^^^^

Randomly zero out elements.

The :class:`.Dropout` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import Dropout

.. autoclass:: UQpy.scientific_machine_learning.activations.Dropout
    :members: forward

______

Dropout 1D
^^^^^^^^^^

Randomly zero out entire 1D feature maps.

The :class:`.Dropout1d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import Dropout1d

.. autoclass:: UQpy.scientific_machine_learning.activations.Dropout1d
    :members: forward

______

Dropout 2D
^^^^^^^^^^

Randomly zero out entire 2D feature maps.

The :class:`.Dropout2d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import Dropout2d

.. autoclass:: UQpy.scientific_machine_learning.activations.Dropout2d
    :members: forward

______

Dropout 3D
^^^^^^^^^^

Randomly zero out entire 3D feature maps.

The :class:`.Dropout3d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import Dropout3d

.. autoclass:: UQpy.scientific_machine_learning.activations.Dropout3d
    :members: forward
