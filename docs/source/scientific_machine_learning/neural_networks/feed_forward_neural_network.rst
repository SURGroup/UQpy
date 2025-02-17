Feed Forward Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.FeedForwardNeuralNetwork` class does not perform any internal computation.
It simply passes tensors along to the network used during initialization.
It is designed to assemble and control Bayesian and ProbabilisticDropout layers in a large neural network via its
:py:meth:`sample` and :py:meth:`drop` methods.

The following example sets the sampling mode in all the :py:class:`BayesianLinear` and dropping mode in
all :py:class:`ProbabilisticDropout` layers with a single call to ``model.sample()`` and ``model.drop()``.

.. literalinclude:: feed_forward_example.txt
    :language: python
    :emphasize-lines: 15, 16
    :linenos:

The :class:`.FeedForwardNeuralNetwork` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import FeedForwardNeuralNetwork


Methods
-------

.. autoclass:: UQpy.scientific_machine_learning.neural_networks.FeedForwardNeuralNetwork
    :members: forward, summary, count_parameters, drop, sample, is_deterministic, set_deterministic

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.FeedForwardNeuralNetwork.network
.. autoattribute:: UQpy.scientific_machine_learning.FeedForwardNeuralNetwork.dropping
.. autoattribute:: UQpy.scientific_machine_learning.FeedForwardNeuralNetwork.sampling