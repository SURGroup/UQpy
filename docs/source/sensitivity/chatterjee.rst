Chatterjee indices
----------------------------------------

The Chatterjee index measures the strength of the relationship between :math:`X` and :math:`Y` using rank statistics :cite:`Chatterjee`.

Consider :math:`n` samples of random variables :math:`X` and :math:`Y`, with :math:`(X_{(1)}, Y_{(1)}), \ldots,(X_{(n)}, Y_{(n)})` such that :math:`X_{(1)} \leq \cdots \leq X_{(n)}`. Here, random variable :math:`X` can be one of the inputs of a model and :math:`Y` be the model response. If :math:`X_{i}`'s have no ties, there is a unique way of doing this (case of ties is also taken into account in the implementation, see :cite:`Chatterjee`). Let :math:`r_{i}`` be the rank of :math:`Y_{(i)}`, that is, the number of :math:`j` such that :math:`Y_{(j)} \leq Y_{(i)}`.The Chatterjee index :math:`\xi_{n}(X, Y)` is defined as:

.. math::

   \xi_{n}(X, Y):=1-\frac{3 \sum_{i=1}^{n-1}\left|r_{i+1}-r_{i}\right|}{n^{2}-1}

The Chatterjee index converges for :math:`n \rightarrow \infty` to the Cramér-von Mises index and is faster to estimate than using the Pick and Freeze approach to compute the the Cramér-von Mises index.

Furthermore, the Sobol indices can be efficiently estimated by leveraging the same rank statistics, which has the advantage that any sample can be used and no specific pick and freeze scheme is required. 

Chatterjee Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ChatterjeeSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.ChatterjeeSensitivity import ChatterjeeSensitivity

Methods
"""""""
.. autoclass:: UQpy.sensitivity.ChatterjeeSensitivity
   :members: run, compute_chatterjee_indices, rank_analog_to_pickfreeze, rank_analog_to_pickfreeze_vec, compute_Sobol_indices

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.ChatterjeeSensitivity.first_order_chatterjee_indices
.. autoattribute:: UQpy.sensitivity.ChatterjeeSensitivity.first_order_sobol_indices
.. autoattribute:: UQpy.sensitivity.ChatterjeeSensitivity.confidence_interval_chatterjee
.. autoattribute:: UQpy.sensitivity.ChatterjeeSensitivity.n_variables
.. autoattribute:: UQpy.sensitivity.ChatterjeeSensitivity.n_samples

Examples
""""""""""

.. toctree::

   Chatterjee Examples <../auto_examples/sensitivity/chatterjee/index>
