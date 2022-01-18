Monte Carlo Sampling
--------------------

The :class:`.MonteCarloSampling` class generates random samples from a specified probability distribution(s).
The :class:`.MonteCarloSampling` class utilizes the :class:`.Distribution` class to define probability distributions.
The advantage of using the :class:`.MonteCarloSampling` class for :py:mod:`UQpy` operations, as opposed to simply generating samples
with the :py:mod:`scipy.stats` package, is that it allows building an object containing the samples and their distributions
for integration with other :py:mod:`UQpy` modules.

Monte Carlo Sampling Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.sampling.MonteCarloSampling
    :members: run, transform_u01

Attributes
~~~~~~~~~~~~~~~~~~
.. autoattribute:: UQpy.sampling.MonteCarloSampling.samples
.. autoattribute:: UQpy.sampling.MonteCarloSampling.samplesU01



.. toctree::
   :hidden:

   Examples <../auto_examples/sampling/monte_carlo/index>
