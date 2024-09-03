Losses
======

This submodules consists of loss functions and divergences used in calculating the error of a neural network model.
Loss functions are typically used to define a "distance" (used colloquially, not as a formal metric)
between a prediction and true value from a dataset. Typically, loss functions map two tensors to a scalar.

In contrast, divergences define a "distance" between a prior and posterior distribution during the training of a
Bayesian neural network. These usually map a neural network
(more specifically, a representation of the distribution of weights) to a scalar.
While both terms contribute to the total loss during training, they are used in very different ways.

For most use cases, we recommend the corresponding Module of the same name rather than calling these functions directly.
For example, using the Module :py:class:`GaussianKullbackLeiblerDivergence` is preferred over calling the function
:func:`gaussian_kullback_leibler_divergence`. If you do need to import these functions, we recommend the following
import statements to prevent naming conflicts with :py:class:`torch.nn.functional`.

>>> import torch.nn.functional as F
>>> import UQpy.scientific_machine_learning.functional as func

-----

Gaussian Kullback-Leibler Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`gaussian_kullback_leibler_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import gaussian_kullback_leibler_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.gaussian_kullback_leibler_divergence

-----

Monte Carlo Kullback-Leibler Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`mc_kullback_leibler_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import mc_kullback_leibler_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.mc_kullback_leibler_divergence

-----

Generalized Jensen-Shannon Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`generalized_jensen_shannon_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import generalized_jensen_shannon_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.generalized_jensen_shannon_divergence

-----

Geometric Jensen-Shannon Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`geometric_jensen_shannon_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import geometric_jensen_shannon_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.geometric_jensen_shannon_divergence


