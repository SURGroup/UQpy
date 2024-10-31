Neural Network Trainers
-------------------------

The neural network trainers are convenient way to loop over training data and learn the parameters of a model.
Trainers are not required as all neural networks in this module can be trained with a user-defined training scheme.
That said, we recommend first-time Bayesian users train their models using :code:`BBBTrainer` to appropriately handle
the divergence calculations.

All trainers require the user to define a model, optimization algorithm using PyTorch, and training data.
Additional options are included to implement schedulers, divergences for Bayesian networks, and
controls over the training / testing behavior.

These trainers are useful in small examples throughout this documentation and robust in practical application.
The :code:`Trainer` is analogous to many of the training functions in the PyTorch documentation.
The Bayes-by-backprop :cite:`blundell2015weight` :code:`BBBTrainer` and Hamiltonian Monte Carlo :cite:`neal2011hmc` :code:`HMCTrainer` trainers
are specific to Bayesian neural networks.

List of Trainers
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Trainer <trainer>
    BBBTrainer <bbb_trainer>
    HMCTrainer <hmc_trainer>
