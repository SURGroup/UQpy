Neural Network Baseclass
------------------------

This is the parent class to all neural networks.
The :class:`NeuralNetwork` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork
    :members: summary, count_parameters, drop, sample, is_deterministic

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork.dropping
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork.sampling