U-Shaped Neural Network
~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.UNeuralNetwork` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import UNeuralNetworkSequential


Methods
-------

.. autoclass:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetworkSequential
    :members: forward, encode, decode, construct_encoder, construct_decoder

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.FeedForwardNeuralNetwork.dropping
.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.FeedForwardNeuralNetwork.sampling

The U-Shaped neural network has one additional class attribute for each encoder and decoder named :code:`encoder_0`,
:code:`decoder_0`, :code:`encoder_1`, :code:`decoder_1`, ..., up to :code:`len(filter_sizes)-1`.
