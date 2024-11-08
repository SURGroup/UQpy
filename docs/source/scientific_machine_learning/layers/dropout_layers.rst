List of Probabilistic Dropout Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All dropout layers are nearly identical implementations to their counterparts in Pytorch, based on the work by Gal et al :cite:`gal2016dropout`.
The difference is these class have a ``dropping`` attribute that controls whether or not they are
active, rather than relying on the ``training`` attribute as Pytorch's implementations do.

This allows us to more conveniently call the dropout methods on forward calls of a neural network,
which is helpful for various computations in uncertainty quantification.

Probabilistic Dropout
^^^^^^^^^^^^^^^^^^^^^

Randomly zero out elements.

The :class:`.ProbabilisticDropout` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import ProbabilisticDropout

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticDropout
    :members: forward

______

Probabilistic Dropout 1d
^^^^^^^^^^^^^^^^^^^^^^^^

Randomly zero out entire 1d feature maps.

The :class:`.ProbabilisticDropout1d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import ProbabilisticDropout1d

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticDropout1d
    :members: forward

______

Probabilistic Dropout 2d
^^^^^^^^^^^^^^^^^^^^^^^^

Randomly zero out entire 2d feature maps.

The :class:`.ProbabilisticDropout2d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import ProbabilisticDropout2d

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticDropout2d
    :members: forward

______

Probabilistic Dropout 3d
^^^^^^^^^^^^^^^^^^^^^^^^

Randomly zero out entire 3d feature maps.

The :class:`.ProbabilisticDropout3d` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import ProbabilisticDropout3d

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticDropout3d
    :members: forward

______

Examples
^^^^^^^^

.. toctree::

    Probabilistic Dropout Examples <../../auto_examples/scientific_machine_learning/mcd_trainer/index>