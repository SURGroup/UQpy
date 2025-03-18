Hamiltonian Monte Carlo Trainers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UQpy does not have its own implementation of a Hamiltonian Monte Carlo (HMC) trainer.
For using the HMC algorithm to train neural networks, we refer users to Pyro HMC (https://docs.pyro.ai/en/stable/mcmc.html#hmc)
or Hamiltorch HMC (https://adamcobb.github.io/journal/hamiltorch.html). Both of these packages are based on PyTorch,
although it may be simpler to implement ``UQpy.sml`` models Hamiltorch.

Examples
--------

.. toctree::

    FNO with Hamiltorch <../../auto_examples/scientific_machine_learning/hmc_trainer/burgers_fno>

