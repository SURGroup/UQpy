Neural Network Baseclass
------------------------

This is the parent class to all neural networks.
The :class:`NeuralNetwork` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

The documentation in the :py:meth:`forward` on this page may be inherited from PyTorch docstrings.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork
    :members: forward, summary, count_parameters, drop, sample, is_deterministic, set_deterministic

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork.dropping
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NeuralNetwork.sampling