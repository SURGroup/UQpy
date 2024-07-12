Scientific Machine Learning
===========================

This module contains functionality for Scientific Machine Learning methods supported in :py:mod:`UQpy`.
This package focuses on supervised machine learning, specifically on the architecture and training of Neural Networks.

This module is *not* intended as a standalone package for neural networks.
It is designed as an extension of `pytorch <https://pytorch.org/>`_ and, as much as practical,
we borrow their syntax and notation to implement UQ methods in a compatible way.
For example, the Bayesian counterpart of torch's ``Linear`` layer is UQpy's ``BayesianLinear``,
which uses similar inputs.

.. figure:: ./figures/uq4ml.png
   :align: center
   :class: with-border
   :width: 400
   :alt: A diagram showing uncertainty quantification improving machine learning, and machine learning informing uncertainty quantification.

   The relationship between UQ4ML and ML4UQ.

The module contains the following parent classes for neural networks:

- :class:`ActivationFunction`: Parent class to all activation functions. Subclass of :class:`torch.nn.Module`.
- :class:`BayesianLayer`: Parent class to all Bayesian layers. Subclass of :class:`Layer`
- :class:`.Layer`: Parent class to all Neural Network Layers. Subclass of :class:`torch.nn.Module`.
- :class:`.Loss`: Parent class to all Loss functions. Subclass of :class:`torch.nn.Module`.
- :class:`.NeuralNetwork`: Parent class to all Neural Networks and Neural Operators.  Subclass of :class:`torch.nn.Module`.

Consistent with Pytorch's architecture, those classes and their subclasses primarily organize and store tensors and parameters.
The bulk of the computation is done by the methods in the ``functional`` folder.
For example, the ``GaussianKullbackLeiblerDivergence`` class is used to define the divergence for a Bayesian neural
network, but hands off the actual divergence computation ``functional.gaussian_kullback_leibler_divergence``.

To prevent naming conflicts with the common torch imports, we recommend importing this module as the following.

>>> import torch.nn as nn
>>> import torch.nn.functional as F
>>> import UQpy.scientific_machine_learning as sml  # jokingly pronounced 'smile'
>>> import UQpy.scientific_machine_learning.functional as func


Getting Started
---------------

.. toctree::

    Bayesian Quickstart Training <../auto_examples/scientific_machine_learning/bayesian_quickstart/bayesian_quickstart_training>
    Bayesian Quickstart Testing <../auto_examples/scientific_machine_learning/bayesian_quickstart/bayesian_quickstart_testing>


Documentation
-------------

.. toctree::
   :maxdepth: 2

    Activations <activations/index>
    Functional <functional/index>
    Layers <layers/index>
    Loss Functions <losses>
    Neural Networks <neural_networks/index>
    Trainers <trainers/index>
