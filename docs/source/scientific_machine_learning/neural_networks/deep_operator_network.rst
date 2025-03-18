Deep Operator Network (DeepONet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of the Deep Operator Network (DeepONet) as defined by Lu et al :cite:`lu2021deeponet`.
The deep operator network is defined by its branch and trunk networks, as shown below.

.. figure:: ./figures/deep_operator_network_diagram.png
   :align: center
   :class: with-border
   :width: 600
   :alt: A diagram showing the architecture of a deep operator network.

   The architecture of a generic deep operator network.

The :class:`.DeepOperatorNetwork` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import DeepOperatorNetwork


Methods
-------

.. autoclass:: UQpy.scientific_machine_learning.neural_networks.DeepOperatorNetwork
    :members: forward, summary, count_parameters, drop, sample, is_deterministic, set_deterministic

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.DeepOperatorNetwork.branch_network
.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.DeepOperatorNetwork.trunk_network

Examples
--------

.. toctree::

   All Examples <../../auto_examples/scientific_machine_learning/deep_operator_network/index>
   Integral 1D <../../auto_examples/scientific_machine_learning/deep_operator_network/integral_1d>
   Linear Elastic <../../auto_examples/scientific_machine_learning/deep_operator_network/linear_elastic>
   Bayesian Linear Elastic <../../auto_examples/scientific_machine_learning/deep_operator_network/bayesian_linear_elastic>