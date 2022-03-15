PCE sensitivity method
----------------------------------------

**Variance-based sensitivity analysis** is a form of a global sensitivity analysis. The **Sobol indices** (Sobol, 1993) allow
the decomposition of the variance of the model output into a sum of contributions associated to individual inputs or
sets of inputs. These contributions can be considered as measures of sensitivity of input parameters and can also
measure the effect of interactions between them.

Traditionally, Sobol indices are computed with Monte Carlo simulations which can be CPU-intensive for high-fidelity and
computationally expensive models. Alternatively, Sudret (2008) proposed a simple post-processing of the coefficients of a
polynomial chaos expansion (PCE) surrogate to compute sensitivity indices.

Let us consider a model :math:`Y = \mathcal{M}(x)` which generates a **scalar quantity of interest (QoI)**, with :math:`Y \in \mathbb{R}` and a random vector
with independent components :math:`X \in \mathbb{R}^L` described by the joint probability density function :math:`f_X`.
A PCE surrogate can be constructed as

.. math:: Y = \mathcal{M}(x) = \sum_{\mathbf{k} \in \Lambda} y_{\mathbf{k}} \Psi_{\mathbf{k}} (X)

where the :math:`\Psi_{\mathbf{k}}(X)` are multivariate polynomials orthonormal with respect to :math:`f_X` and
:math:`y_{\mathbf{k}} \in \mathbb{R}` are the corresponding coefficients. the multi-indices
:math:`\mathbf{k} = \left(k_1, \dots, k_N\right)` are uniquely associated
to the single indices :math:`k`, and :math:`\Lambda` is a multi-index set with cardinality :math:`\#\Lambda = K`.
In :py:mod:`UQpy`, a PCE surrogate can be constructed using the class :class:`.PolynomialChaosExpansion`.

The PCE coefficients in :math:`\Lambda` can be interpreted as partial variances due to specific RV interactions
defined by the multi-indices :math:`\mathbf{k}`. The multi-indices corresponding to partial variances caused by :math:`X_n`,
either individually (**first-order**) or in combination with the remaining RV (**total-effect**), can then be collected into
the multi-index sets :math:`\Lambda_n^\text{F} \subset \Lambda` and :math:`\Lambda_n^\text{T} \subset \Lambda`, respectively, defined as

.. math:: \Lambda_n^{\text{F}} &= \{\mathbf{k} \in \Lambda \; : \; k_n \neq 0 \:\: \text{and} \:\: k_l = 0, l \neq n\}, \\
.. math:: \Lambda_n^{\text{T}} &= \{\mathbf{k} \in \Lambda \; : \; k_n \neq 0\}. \\

The corresponding first-order and total-effect Sobol indices are then estimated as

.. math:: S_n^{\text{F}} &\approx \frac{\sum_{\mathbf{k} \in \Lambda_n^{\text{F}}} y_{\mathbf{k}}^2}{\sum_{\mathbf{k} \in \Lambda \setminus \mathbf{0}} y_{\mathbf{k}}^2}, \\
.. math:: S_n^{\text{T}} &\approx \frac{\sum_{\mathbf{k} \in \Lambda_n^{\text{T}}} y_{\mathbf{k}}^2}{\sum_{\mathbf{k} \in \Lambda \setminus \mathbf{0}} y_{\mathbf{k}}^2}. \\

In case of a **multivariate QoI** :math:`\mathbf{Y} = \mathcal{M}(x)`, the procedure presented above can be applied element-wise
to compute Sobol indices for each QoI component :math:`Y_m`, :math:`m=1,\dots,M`.
Global sensitivity analysis based on the covariance decomposition approach requires the computation of the traces of
the covariance matrices with respect to the components :math:`Y_m`, which depend on specific combinations of the input RV.
These traces are equal to the sum of the variances of all QoI components dependent on the specific input RV combinations.
These partial variances are easily obtained from the PCE coefficients and the **generalized sensitivity indices** :math:`G_n^{\text{F}}`
and :math:`G_n^{\text{T}}`, can be estimated as

.. math:: G_n^{\text{F}} &\approx \frac{\sum_{m=1}^M \left(\sum_{\mathbf{k} \in \Lambda_{m,n}^{\text{F}}} y_{m,\mathbf{k}}^2\right)}{\sum_{m=1}^M \left(\sum_{\mathbf{k} \in \Lambda_m \setminus \mathbf{0}} y_{m,\mathbf{k}}^2\right)},\\
.. math:: G_n^{\text{T}} &\approx \frac{\sum_{m=1}^M \left(\sum_{\mathbf{k} \in \Lambda_{m,n}^{\text{T}}} y_{m,\mathbf{k}}^2\right)}{\sum_{m=1}^M \left(\sum_{\mathbf{k} \in \Lambda_m \setminus \mathbf{0}} y_{m,\mathbf{k}}^2\right)}. \\

Therefore, once a PCE approximation of a multivariate QoI is available, the generalized sensitivity indices and can be estimated with negligible computational cost by simply post-processing the PCE terms, similar to the case of a scalar QoI.


PCE Sensitivity class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.PceSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.PceSensitivity import PceSensitivity

.. autoclass:: UQpy.sensitivity.PceSensitivity
    :members:

Methods
"""""""
.. autoclass:: UQpy.sensitivity.PceSensitivity
    :members:

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.PceSensitivity.first_order_indices
.. autoattribute:: UQpy.sensitivity.PceSensitivity.total_order_indices

Examples
""""""""""

.. toctree::

   Morris Examples <../auto_examples/sensitivity/pce/index>
