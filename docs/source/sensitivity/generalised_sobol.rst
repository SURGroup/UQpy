Generalised Sobol indices
----------------------------------------

A natural generalization of the Sobol indices (that are classically defined for single-output models) for multi-output models. The generalised Sobol indices are computed using the Pick-and-Freeze approach. (For implementation details, see also [1]_.)

Consider a model :math:`Y=f(X): \mathbb{R}^d \rightarrow \mathbb{R}^k` with :math:`d` inputs :math:`X=\left[ X_{1}, X_{2},…,X_{d} \right]` and :math:`k` outputs :math:`Y=\left[ Y_{1}, Y_{2},…,Y_{k} \right]`.

As the inputs :math:`X_{1}, \ldots, X_{d}` are independent, :math:`f` may be decomposed through the so-called Hoeffding decomposition:

.. math::
         f(X) = c + f_{\mathbf{u}}\left(X_{\mathbf{u}}\right)+f_{\sim \mathbf{u}}\left(X_{\sim \mathbf{u}}\right) + f_{\mathbf{u}, \sim \mathbf{u}}\left(X_{\mathbf{u}}, X_{\sim \mathbf{u}}\right)

where :math:`c \in \mathbb{R}^{k}, f_{\mathbf{u}}: E_{\mathbf{u}} \rightarrow \mathbb{R}^{k}, f_{\sim \mathbf{u}}: E_{\sim \mathbf{u}} \rightarrow \mathbb{R}^{k}` and :math:`f_{\mathbf{u}, \sim \mathbf{u}}: E \rightarrow \mathbb{R}^{k}` are given by

.. math::
   c = \mathbb{E}(Y), 
   
.. math::
   f_{\mathbf{u}}=\mathbb{E}\left(Y \mid X_{\mathbf{u}}\right)-c, 
   
.. math::   
   f_{\sim \mathbf{u}}=\mathbb{E}\left(Y \mid X_{\sim \mathbf{u}}\right)-c, 
   
.. math::
   f_{u, \sim \mathbf{u}}=Y-f_{\mathbf{u}}-f_{\sim \mathbf{u}}-c.

Thanks to :math:`L^{2}`-orthogonality, computing the covariance matrix of both sides of the above equation leads to

.. math::
   \Sigma = C_{\mathbf{u}}+C_{\sim \mathbf{u}}+C_{\mathbf{u}, \sim \mathbf{u}}.

Here, :math:`\Sigma, C_{\mathbf{u}}, C_{\sim \mathbf{u}}` and :math:`C_{\mathbf{u}, \sim \mathbf{u}}` are denoting the covariance matrices of :math:`Y, f_{\mathbf{u}}\left(X_{\mathbf{u}}\right), f_{\sim \mathbf{u}}\left(X_{\sim \mathbf{u}}\right)` and :math:`f_{\mathbf{u}, \sim \mathbf{u}}\left(X_{\mathbf{u}}, X_{\sim \mathbf{u}}\right)` respectively.

The first order generalised Sobol indices can be computed using the Pick-and-Freeze approach as follows, where :math:`\mathbf{u}` is a variable :math:`i` of the independent random variables.

.. math::
   S_{i, N}=\frac{\operatorname{Tr}\left(C_{i, N}\right)}{\operatorname{Tr}\left(\Sigma_{N}\right)}

where :math:`C_{\mathbf{i}, N}` and :math:`\Sigma_{N}` are the empirical estimators of :math:`C_{\mathbf{i}}=\operatorname{Cov}\left(Y, Y^{\mathbf{i}}\right)` and :math:`\Sigma=\mathbb{V}[Y]` defined by

.. math::
   C_{\mathbf{i}, N}=\frac{1}{N} \sum_{j=1}^{N} Y_{j}^{\mathrm{i}} Y_{j}^{t}-\left(\frac{1}{N} \sum_{j=1}^{N} \frac{Y_{j}+Y_{j}^{\mathbf{i}}}{2}\right)\left(\frac{1}{N} \sum_{j=1}^{N} \frac{Y_{j}+Y_{j}^{\mathbf{i}}}{2}\right)^{t}

and

.. math::
   \Sigma_{N}=\frac{1}{N} \sum_{j=1}^{N} \frac{Y_{j} Y_{j}^{t}+Y_{j}^{\mathbf{i}}\left(Y_{j}^{\mathbf{i}}\right)^{t}}{2}-\left(\frac{1}{N} \sum_{j=1}^{N} \frac{Y_{j}+Y_{j}^{\mathbf{i}}}{2}\right)\left(\frac{1}{N} \sum_{j=1}^{N} \frac{Y_{j}+Y_{j}^{\mathbf{i}}}{2}\right)^{t}


.. [1] Gamboa F, Janon A, Klein T, Lagnoux A, others. Sensitivity analysis for multidimensional and functional outputs. Electronic journal of statistics 2014; 8(1): 575-603.(`Link <https://hal.inria.fr/hal-00881112/document>`_)


Generalised Sobol Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Generalised Sobol` class is imported using the following command:

>>> from UQpy.sensitivity.generalised_sobol import GeneralisedSobol

Methods
"""""""

.. autoclass:: UQpy.sensitivity.GeneralisedSobol
     :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.GeneralisedSobol.gen_sobol_i
.. autoattribute:: UQpy.sensitivity.GeneralisedSobol.gen_sobol_total_i
.. autoattribute:: UQpy.sensitivity.GeneralisedSobol.n_samples
.. autoattribute:: UQpy.sensitivity.GeneralisedSobol.num_vars

Examples
""""""""""

.. toctree::

   Generalised Sobol Examples <../auto_examples/sensitivity/generalised_sobol/index>