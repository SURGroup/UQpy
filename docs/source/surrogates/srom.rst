
Stochatic Reduced Order Models - SROMs
----------------------------------------

A SROM is a sample-based surrogate for probability models. An SROM takes a set of samples and attributes of a distribution and optimizes the sample probability weights according to the method in :cite:`Surrogates1`. More specifically, an SROM constructs a reduce order model for arbitrary random variables `X` as follows.

.. math:: \tilde{X} =  \begin{cases} x_1 & probability \text{  }p_1^{(opt)} \\ & \vdots \\ x_m & probability \text{  }p_m^{(opt)} \end{cases}

where :math:`\tilde{X}` is defined by an arbitrary set of samples :math:`x_1, \dots, x_m` defined over the same support as :math:`X` (but not necessarily drawn from its probability distribution) and their assigned probability weights. The probability weights are defined such that the total error between the sample empirical probability distribution, moments and correlation of :math:`\tilde{X}` and those of the random variable `X` is minimized. This optimization problem can be express as:

.. math:: & \min_{\mathbf{p}}  \sum_{u=1}^3 \alpha_u e_u(\mathbf{p}) \\ & \text{s.t.} \sum_{k=1}^m p_k =1 \quad and \quad p_k \geq 0, \quad k=1,2,\dots,m

where :math:`\alpha_1`, :math:`\alpha_2`, :math:`\alpha_3 \geq 0` are constants defining the relative importance of the marginal distribution, moments and correlation error between the reduce order model and actual random variables in the objective function.

.. math:: &  e_{1}(p)=\sum\limits_{i=1}^d \sum\limits_{k=1}^m w_{F}(x_{k,i};i)(\hat{F}_{i}(x_{k,i})-F_{i}(x_{k,i}))^2  \\ & e_{2}(p)=\sum\limits_{i=1}^d \sum\limits_{r=1}^2 w_{\mu}(r;i)(\hat{\mu}(r;i)-\mu(r;i))^2 \\ & e_{3}(p)=\sum\limits_{i,j=1,...,d ; j>i}  w_{R}(i,j)(\hat{R}(i,j)-R(i,j))^2

Here, :math:`F(x_{k,i})` and :math:`\hat{F}(x_{k,i})` denote the marginal cumulative distributions of :math:`\mathbf{X}` and :math:`\mathbf{\tilde{X}}` (reduced order model) evaluated at point :math:`x_{k,i}`, :math:`\mu(r; i)` and :math:`\hat{\mu}(r; i)` are the marginal moments of order `r` for variable `i`, and :math:`R(i,j)` and :math:`\hat{R}(i,j)` are correlation matrices of :math:`\mathbf{X}` and :math:`\mathbf{\tilde{X}}` evaluted for components :math:`x_i` and :math:`x_j`. Note also that `m` is the number of sample points and `d` is the number of random variables.

SROM Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SRMO` class is imported using the following command:

>>> from UQpy.surrogates.stochastic_reduced_order_models.SROM import SROM

Methods
"""""""
.. autoclass:: UQpy.surrogates.stochastic_reduced_order_models.SROM
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.surrogates.stochastic_reduced_order_models.SROM.sample_weights


Examples
""""""""""

.. toctree::

   Stochastic Reduced Order Model Examples <../auto_examples/surrogates/srom/index>