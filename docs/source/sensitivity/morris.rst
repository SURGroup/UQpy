Morris method
----------------------------------------

The Morris method is a so-called one-at-a-time (OAT) screening method that is known to achieve a good trade-off between
accuracy and efficiency, and can serve to identify the few important factors in models with many factors. The method is
based on calculating for each input a number of incremental ratios, called Elementary Effects (EE), from which basic
statistics are computed to derive sensitivity information :cite:`Morris1`.

For each input :math:`X_{k}`, the elementary effect is computed as:

.. math::  EE_{k} = \frac{Y(X_{1}, ..., X_{k}+\Delta, ..., X_{d})-Y(X_{1}, ..., X_{k}, ..., X_{d})}{\Delta}

where :math:`\Delta` is chosen so that :math:`X_{k}+\Delta` is still in the allowable domain for every dimension k.

The key idea of the original Morris method (current implementation) is to initiate trajectories from various “nominal” points :math:`X` randomly selected over the grid and then gradually advancing one :math:`\Delta` at a time between each model evaluation (one at a time OAT design), along a different dimension of the parameter space selected randomly (see an example of 5 trajectories in a 2D space on the left plot of the figure below). For :math:`r` trajectories (usually set :math:`r` between 5 and 50), the number of simulations required is :math:`r (d+1)`.

The following sensitivity indices are computed from the elementary effects:

.. math:: \mu_{k}^{\star} = \frac{1}{r} \sum_{i=1}^{r} \vert EE_{k}^{r} \vert

.. math:: \sigma_{k} = \sqrt{ \frac{1}{r} \sum_{i=1}^{r} \left( EE_{k}^{r} - \mu_{k} \right)^{2}}

These indices allow differentiation between three groups of inputs (see example in the right plot on the figure below):

* Parameters with non-influential effects, i.e., the parameters that have relatively small values of both :math:`\mu_{k}^{\star}` and :math:`\sigma_{k}`.
* Parameters with linear and/or additive effects, i.e., the parameters that have a relatively large value of :math:`\mu_{k}^{\star}` and relatively small value of :math:`\sigma_{k}` (the magnitude of the effect :math:`\mu_{k}^{\star}` is consistently large, regardless of the other parameter values, i.e., no interaction).
* Parameters with nonlinear and/or interaction effects, i.e., the parameters that have a relatively small value of :math:`\mu_{k}^{\star}` and a relatively large value of :math:`\sigma_{k}` (large value of :math:`\sigma_{k}` indicates that the effect can be large or small depending on the other values of parameters at which the model is evaluated, indicates potential interaction between parameters).

.. image:: ../_static/morris_indices.png
   :align: left


Morris Class
^^^^^^^^^^^^^^^^^^

The :class:`.MorrisSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.MorrisSensitivity import MorrisSensitivity

Methods
"""""""
.. autoclass:: UQpy.sensitivity.MorrisSensitivity
    :members: run, sample_trajectories

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.MorrisSensitivity.trajectories_unit_hypercube
.. autoattribute:: UQpy.sensitivity.MorrisSensitivity.trajectories_physical_space
.. autoattribute:: UQpy.sensitivity.MorrisSensitivity.elementary_effects
.. autoattribute:: UQpy.sensitivity.MorrisSensitivity.mustar_indices
.. autoattribute:: UQpy.sensitivity.MorrisSensitivity.sigma_indices

Examples
""""""""""

.. toctree::

   Morris Examples <../auto_examples/sensitivity/morris/index>


.. Note::
    **Subclassing the Morris class.**
    The user can subclass the Morris class to implement algorithms with better sampling of the trajectories for instance.
    In order to do so, the user can simply overwrite the :py:meth:`sample_trajectories` method, which should take as inputs the
    number of trajectories `trajectories_number` and any other user-defined input (transferred from the :py:meth:`run` method as `kwargs`).
