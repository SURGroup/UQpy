Morris Screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a model of the sort :math:`Y=h(X)`, :math:`Y` is assumed to be scalar, :math:`X=[X_{1}, ..., X_{d}]`.

For each input ;math:`X_{k}`, the elementary effect is computed as:

.. math:: EE_{k} = \frac{Y(X_{1}, ..., X_{k}+\Delta, ..., X_{d})-Y(X_{1}, ..., X_{k}, ..., X_{d})}{\Delta}

where :math:`\Delta` is chosen so that :math:`X_{k}+\Delta` is still in the allowable domain for every dimension :math:`k`.

The key idea of the original Morris method is to initiate trajectories from various “nominal” points X randomly
selected over the grid and then gradually advancing one :math:`\Delta` at a time between each model evaluation
(one at a time OAT design), along a different dimension of the parameter space selected randomly. For :math:`r` trajectories
(usually set :math:`r` between 5 and 50), the number of simulations required is :math:`r (d+1)`.

Sensitivity indices are computed as:

.. math:: \mu_{k}^{\star} = \frac{1}{r} \sum_{i=1}^{r} \vert EE_{k}^{r} \vert


.. math:: \sigma_{k} = \sqrt{ \frac{1}{r} \sum_{i=1}^{r} \left( EE_{k}^{r} - \mu_{k} \right)^{2}}


It allows differentiation between three groups of inputs:
- Parameters with non-influential effects, i.e., the parameters that have relatively small values of both
:math:`\mu_{k}^{\star}` and :math:`\sigma_{k}`.
- Parameters with linear and/or additive effects, i.e., the parameters that have a relatively large value of
:math:`\mu_{k}^{\star}` and relatively small value of :math:`\sigma_{k}` (the magnitude of the effect
:math:`\mu_{k}^{\star}` is consistently large, regardless of the other parameter values, i.e., no interaction).
- Parameters with nonlinear and/or interaction effects, i.e., the parameters that have a relatively small value of
:math:`\mu_{k}^{\star}` and a relatively large value of :math:`\sigma_{k}` (large value of :math:`\sigma_{k}` indicates that the
effect can be large or small depending on the other values of parameters at which the model is evaluated,
ndicates potential interaction between parameters).
