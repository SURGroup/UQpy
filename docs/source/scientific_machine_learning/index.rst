Scientific Machine Learning
===========================

This module contains functionality for Scientific Machine Learning methods supported in :py:mod:`UQpy`.
This module is designed for compatibility with `pytorch <https://pytorch.org/>`_.
This package focuses on supervised machine learning, specifically on the architecture and training of Neural Networks.


The module contains the following parent classes:

- :class:`.Layer`: Parent class to all Neural Network Layers. Subclass of :class:`torch.nn.Module`.
- :class:`.Loss`: Parent class to all Loss functions. Subclass of :class:`torch.nn.Module`.
- :class:`.NeuralNetwork`: Parent class to all Neural Networks and Neural Operators.  Subclass of :class:`torch.nn.Module`.


.. toctree::
   :maxdepth: 1
   :caption: Scientific Machine Learning

    Layers <layers>
    Loss Functions <losses>
    Neural Networks <neural_networks/index>
    Trainers <trainers/index>
