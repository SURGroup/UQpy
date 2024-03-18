Scientific Machine Learning
===========================

This module contains functionality for Scientific Machine Learning methods supported in :py:mod:`UQpy`.
This module is design for compatibility with `pytorch <https://pytorch.org/>`_.

The module contains the following parent classes:

- :class:`.Layer`: Parent class to all Neural Network Layers. Subclass of :class:`torch.nn.Module`.
- :class:`.Loss`: Parent class to all Loss functions. Subclass of :class:`torch.nn.Module`
- :class:`.NeuralNetwork`: Parent class to all Neural Networks and Neural Operators.  Subclass of :class:`torch.nn.Module`
- :class:`.Optimizer`: Parent class to all optimziation algorithms. Subclass of :class:`torch.optim.Optimizer`


Scientific Machine Learning methodologies are focused on the architecture and training of Neural Networks.

.. toctree::
   :maxdepth: 1
   :caption: Scientific Machine Learning

    Loss Functions <losses>
    Neural Networks <neural_networks/index>
    Neural Network Layers <layers>
    Optimizers <optimizers>
