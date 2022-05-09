Chatterjee indices
----------------------------------------

The Chatterjee index measures the strength of the relationship between :math:`X` and :math:`Y` using rank statistics.

Consider :math:`n` samples of random variables :math:`X` and :math:`Y`, with :math:`(X_{(1)}, Y_{(1)}), \ldots,(X_{(n)}, Y_{(n)})` such that :math:`X_{(1)} \leq \cdots \leq X_{(n)}`. Here, random variable :math:`X` can be one of the inputs of a model and :math:`Y` be the model response. If :math:`X_{i}`'s have no ties, there is a unique way of doing this (case of ties is also taken into account in the implementation, see [1]_). Let :math:`r_{i}`` be the rank of :math:`Y_{(i)}`, that is, the number of :math:`j` such that :math:`Y_{(j)} \leq Y_{(i)}`.The Chatterjee index :math:`\xi_{n}(X, Y)` is defined as:

.. math::

   \xi_{n}(X, Y):=1-\frac{3 \sum_{i=1}^{n-1}\left|r_{i+1}-r_{i}\right|}{n^{2}-1}

The Chatterjee index converges for :math:`n \rightarrow \infty` to the Cramér-von Mises index and is faster to estimate than using the Pick and Freeze approach in the Cramér-von Mises index.

.. [1] Sourav Chatterjee (2021) A New Coefficient of Correlation, Journal of the American Statistical Association, 116:536, 2009-2022, DOI: 10.1080/01621459.2020.1758115 (`Link <https://doi.org/10.1080/01621459.2020.1758115>`_)

Chatterjee Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Chatterjee` class is imported using the following command:

>>> from UQpy.sensitivity.chatterjee import Chatterjee

Methods
"""""""
.. autoclass:: UQpy.sensitivity.Chatterjee
   :members: run, compute_chatterjee_indices, rank_analog_to_pickfreeze, compute_Sobol_indices

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.Chatterjee.chatterjee_i
.. autoattribute:: UQpy.sensitivity.Chatterjee.sobol_i
.. autoattribute:: UQpy.sensitivity.Chatterjee.CI_chatterjee_i
.. autoattribute:: UQpy.sensitivity.Chatterjee.num_vars
.. autoattribute:: UQpy.sensitivity.Chatterjee.n_samples

Examples
""""""""""

.. toctree::

   Chatterjee Examples <../auto_examples/sensitivity/chatterjee/index>
