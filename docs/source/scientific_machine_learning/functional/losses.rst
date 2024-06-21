Losses
======

Documentation for loss functions and divergences.
Note that Loss functions are typically used to define a "distance" (used colloquially, not as a formal metric)
between a prediction and true value from a dataset.

In contrast, divergences define a "distance" between a prior and posterior distribution during the training of a
Bayesian neural network. While both terms contribute to the total loss, they are used in very different ways.


Gaussian Kullback-Leiber Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`gaussian_kullback_leiber_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import gaussian_kullback_leiber_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.gaussian_kullback_leiber_divergence

-----

Gaussian Jenson-Shannon Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :func:`gaussian_jenson_shannon_divergence` is imported using the following command:

>>> from UQpy.scientific_machine_learning.functional import gaussian_jenson_shannon_divergence

.. autofunction:: UQpy.scientific_machine_learning.functional.gaussian_jenson_shannon_divergence
