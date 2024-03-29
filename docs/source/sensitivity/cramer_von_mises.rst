Cramér-von Mises indices
----------------------------------------

A sensitivity index based on the Cramér-von Mises distance. In contrast to the variance based Sobol indices, it takes into account the whole distribution of the model output and is therefore considered as a moment-free method :cite:`CVM`. Furthermore the index can be naturally extended to multivariate model outputs (not implemented yet in UQPy). 

Consider a model :math:`Y=f(X): \mathbb{R}^d \rightarrow \mathbb{R}^k` with :math:`d` inputs :math:`X_{(1)}, X_{(2)}, \ldots, X_{(d)}` and :math:`k` outputs :math:`Y_{(1)}, Y_{(2)}, \ldots, Y_{(k)}`. We define the cumulative distribution function :math:`F(t)` of :math:`Y` as:

.. math::

   F(t)=\mathbb{P}(Z \leqslant t)=\mathbb{E}\left[\mathbb{1}_{\{Z \leqslant t\}}\right] \text { for } t=\left(t_{1}, \ldots, t_{k}\right) \in \mathbb{R}^{k}

and the conditional distribution function :math:`F(t)` of :math:`Y` as:

.. math::

   F^{v}(t)=\mathbb{P}\left(Z \leqslant t \mid X_{v}\right)=\mathbb{E}\left[\mathbb{1}_{\{Z \leqslant t\}} \mid X_{v}\right] \text { for } t=\left(t_{1}, \ldots, t_{k}\right) \in \mathbb{R}^{k}

where, :math:`\{Z \leqslant t\} \text { means that } \left\{Z_{1} \leqslant t_{1}, \ldots, Z_{k} \leqslant t_{k}\right\}`.

The first order Cramér-von Mises index :math:`S_{2, C V M}^{i}` (for input :math:`v = {i}`) is defined as:

.. math::

   S_{2, C V M}^{i}:=\frac{\int_{\mathbb{R}^{k}} \mathbb{E}\left[\left(F(t)-F^{i}(t)\right)^{2}\right] d F(t)}{\int_{\mathbb{R}^{k}} F(t)(1-F(t)) d F(t)}

and the total Cramér-von Mises index :math:`S_{2, C V M}^{T o t, i}` (for input :math:`v = {i}`) is defined as:

.. math::

   S_{2, C V M}^{T o t, i}:=1-S_{2, C V M}^{\sim i}=1-\frac{\int_{\mathbb{R}^{k}} \mathbb{E}\left[\left(F(t)-F^{\sim i}(t)\right)^{2}\right] d F(t)}{\int_{\mathbb{R}^{k}} F(t)(1-F(t)) d F(t)}

The above first and total order indices are estimated using the Pick-and-Freeze approach. This requires :math:`N(d+2)` model evaluations, where :math:`N` is the number of samples. (For implementation details, see also :cite:`gamboa2020global`.)

Cramér-von Mises Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CramerVonMisesSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.CramerVonMisesSensitivity import CramerVonMisesSensitivity

Methods
"""""""
.. autoclass:: UQpy.sensitivity.CramerVonMisesSensitivity
   :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.first_order_CramerVonMises_indices
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.confidence_interval_CramerVonMises
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.first_order_sobol_indices
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.total_order_sobol_indices
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.n_samples
.. autoattribute:: UQpy.sensitivity.CramerVonMisesSensitivity.n_variables


Examples
""""""""""

.. toctree::

   Cramér-von Mises Examples <../auto_examples/sensitivity/cramer_von_mises/index>
