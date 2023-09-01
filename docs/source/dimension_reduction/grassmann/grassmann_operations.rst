Grassmann Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:mod:`UQpy` supports several operations on the Grassmann manifold. These operations are defined as methods of the
:class:`.GrassmannOperations` class described herein.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations

Methods
~~~~~~~~~~~~~~~~~~~~~~
The following sections introduce the methods available in the :class:`.GrassmannOperations` class with a brief
introduction to their theory and their specifications in :py:mod:`UQpy`.

Exponential Map
~~~~~~~~~~~~~~~~~~~~~~

For point :math:`\mathbf{X}` on the Grassmann manifold :math:`\mathcal{G}(p,n)`, we can define the tangent space
:math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` as the set of all matrices :math:`\mathbf{\Gamma}` such that

.. math:: \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)} = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} :\mathbf{\Gamma}^T\mathbf{X}=\mathbf{0}\}


Consider two points :math:`\mathbf{X}` and :math:`\mathbf{Y}` on :math:`\mathcal{G}(p, n)` and the tangent space
:math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` at :math:`\mathbf{X}`. The  geodesic, which refers to the shortest
curve on the manifold connecting the two points, is defined as

.. math:: \mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T

.. math:: \Phi(t)=\mathrm{span}\left[\left(\mathbf{X}\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Phi}(0)=\mathbf{X}` and :math:`\mathbf{\Phi}(1)=\mathbf{Y}`.


The exponential map, denoted as
:math:`\mathrm{exp}_{\mathbf{X}}(\mathbf{\Gamma}): \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}  \rightarrow \mathcal{G}(p, n)`, maps a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` to the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` in the direction :math:`\mathbf{\Gamma}`.


.. math::  \mathrm{exp}_{\mathbf{X}}(\mathbf{\Gamma})\equiv\mathbf{Y}

.. math:: \mathbf{Y} = \mathrm{exp}_{\mathbf{X}}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{X}\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T

The exponential map is implemented in :py:mod:`UQpy` through the static :meth:`.exp_map` method.

To use the :meth:`.exp_map` method, one needs to import the :class:`.GrassmannOperations` class from the
:mod:`UQpy.dimension_reduction.grassmann_manifold` module as follows:

>>> from UQpy.dimension_reduction.grassmann_manifold import GrassmannOperations

Since :meth:`.exp_map` is a static method, it does not require instantiation of the :class:`.GrassmannOperations` class.

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations.exp_map




Logarithmic Map
~~~~~~~~~~~~~~~~~~~~~~

The logarithmic map, denoted as
:math:`\mathrm{log}_\mathbf{X}(\mathbf{Y}):\mathcal{G}(p, n) \rightarrow  \mathcal{T}_{\mathbf{X}\mathcal{G}(p,n)}` maps
the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` to a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}`.


.. math:: \mathrm{log}_\mathbf{X}(\mathbf{Y})\equiv \mathbf{\Gamma} = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T

where :math:`\mathbf{U}, \mathbf{S}, \mathbf{V}` are define as in the exponential map above.

To use the :meth:`.log_map` method, one needs to import the :class:`.GrassmannOperations` class from the
:mod:`UQpy.dimension_reduction.grassmann_manifold` module as follows:

>>> from UQpy.dimension_reduction.grassmann_manifold import GrassmannOperations


Since :meth:`.log_map` is a static method, it does not require instantiation of the :class:`.GrassmannOperations` class.

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations.log_map

Karcher mean
~~~~~~~~~~~~~~~~~~~~~~

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Karcher mean is defined as the
solution of the following minimization problem:

.. math:: \mu = \arg_{\mathbf{Y}} \mathrm{min}\left(\frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i, \mathbf{Y})^2\right)

where :math:`d(\cdot)` is a Grassmann distance measure and :math:`\mathbf{Y}` is a reference point on :math:`\mathcal{G}(p,n)`.

To use the :meth:`.karcher_mean` method, one needs to import the :class:`.GrassmannOperations` class from the
:mod:`UQpy.dimension_reduction.grassmann_manifold` module as follows:

>>> from UQpy.dimension_reduction.grassmann_manifold import GrassmannOperations

Since :meth:`.karcher_mean` is a static method, it does not require instantiation of the :class:`.GrassmannOperations`
class.

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations.karcher_mean

:mod:`UQpy` offers two methods for solving this optimization, the :class:`.GradientDescent` and the
:class:`.StochasticGradientDescent`. Both are implemented as private methods of the :class:`.GrassmannOperations`
class.


Frechet variance
~~~~~~~~~~~~~~~~~~~~~~

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Frechet variance is defined as:

.. math:: \sigma_{f}^2 = \frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i, \mu)^2

where :math:`d(\cdot)` is a Grassmann distance measure and :math:`\mu` is the Karcher mean of set of points :math:`\{\mathbf{X}_i\}_{i=1}^N` on :math:`\mathcal{G}(p,n)`.

To use the :meth:`.frechet_variance` method, one needs to import the :class:`.GrassmannOperations` class from the
:mod:`UQpy.dimension_reduction.grassmann_manifold` module as follows:

>>> from UQpy.dimension_reduction.grassmann_manifold import GrassmannOperations

Since :meth:`.frechet_variance` is a static method, it does not require instantiation of the
:class:`.GrassmannOperations` class.

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations.frechet_variance