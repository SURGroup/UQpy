
Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exponential map, denoted as :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}  \rightarrow \mathcal{G}(p, n)`, maps a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` to the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` in the direction :math:`\mathbf{\Gamma}`.


.. math::  \mathrm{exp}_{\mathbf{X}}(\mathbf{\Gamma})\equiv\mathbf{Y} 

.. math:: \mathbf{Y} = \mathrm{exp}_{\mathbf{X}}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{X}\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T


The :class:`.GradientDescent` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.optimization.GradientDescent import GradientDescent

One can use the following command to instantiate the class :class:`.GradientDescent`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.optimization.GradientDescent
    :members:


Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The logarithmic map, denoted as :math:`\mathcal{G}(p, n) \rightarrow  \mathcal{T}_{\mathbf{X}\mathcal{G}(p,n)}` maps the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` to a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}`.


.. math:: \mathrm{log}_\mathbf{X}(\mathbf{Y})\equiv \mathbf{\Gamma} = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T 

The :class:`.StochasticGradientDescent` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.optimization.StochasticGradientDescent import StochasticGradientDescent

One can use the following command to instantiate the class :class:`.StochasticGradientDescent`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.optimization.StochasticGradientDescent
    :members:

___________________________________________________________________________________________


The abstract :class:`.OptimizationMethod` class is a blueprint for classes in :mod:`.optimization` module. It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.optimization.baseclass.OptimizationMethod
    :members:


