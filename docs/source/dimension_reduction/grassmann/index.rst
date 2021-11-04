Grassmann manifold
--------------------------------

In differential geometry the Grassmann manifold :math:`\mathcal{G}(p, n)` refers to a collection of
:math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space [1]_, [2]_. A point on :math:`\mathcal{G}(p, n)` is typically represented as a :math:`n \times p` orthonormal matrix :math:`\mathbf{X}`, whose column spans the corresponding subspace.  For point :math:`\mathbf{X}` we can define the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` as  the set of all matrices :math:`\mathbf{\Gamma}` at :math:`\mathbf{X}` such as 

.. math:: \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)} = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} : \mathbf{\Gamma}^T\mathbf{X}=\mathbf{0}\}


Consider two points :math:`\mathbf{X}` and :math:`\mathbf{Y}` on :math:`\mathcal{G}(p, n)` and the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` at :math:`\mathbf{X}`. The  geodesic, which refers to the shortest curve on the manifold connecting the two points, is defined as

.. math:: \mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T

.. math:: \Phi(t)=\mathrm{span}\left[\left(\mathbf{X}\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Phi}(0)=\mathbf{X}` and :math:`\mathbf{\Phi}(1)=\mathbf{Y}`.

The :class:`.Grassmann` class offers various methods to operate with data on the Grassmann manifold.


.. toctree::
   :maxdepth: 1
   :caption: Methods

Exponential mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exponential map, denoted as :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}  \rightarrow \mathcal{G}(p, n)`, maps a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` to the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` in the direction :math:`\mathbf{\Gamma}`.


.. math::  \mathrm{exp}_{\mathbf{X}}(\mathbf{\Gamma})\equiv\mathbf{Y} 

.. math:: \mathbf{Y} = \mathrm{exp}_{\mathbf{X}}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{X}\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T


The :class:`.exp_map` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.grassmann import exp_map

One can use the following command to instantiate the class :class:`.exp_map`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.grassmann.exp_map
    :members:
	
	
	

Logarithmic map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The logarithmic map, denoted as :math:`\mathcal{G}(p, n) \rightarrow  \mathcal{T}_{\mathbf{X}\mathcal{G}(p,n)}` maps the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` to a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}`.


.. math:: \mathrm{log}_\mathbf{X}(\mathbf{Y})\equiv \mathbf{\Gamma} = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T 

The :class:`.log_map` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.grassmann import log_map

One can use the following command to instantiate the class :class:`.log_map`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.grassmann.log_map
    :members:


Frechet variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Frechet variance is defined as the solution of the following minimization problem: 

.. math:: \sigma_{f}^2 = \mathrm{min}(\frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i - \mathbf{Y})^2)

where :mat:`d(\cdot)` is a Grassmann distance metric and :math:`\mathbf{Y}` is a reference point on :math:`\mathcal{G}(p,n)`.


The :class:`.frechet_variance` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.grassmann import frechet_variance

One can use the following command to instantiate the class :class:`.frechet_variance`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.grassmann.frechet_variance
    :members:
	

Karcher mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Karcher mean is defined as the solution of the following minimization problem: 

.. math:: \mu = \arg \mathrm{min}(\frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i - \mathbf{Y})^2)

where :mat:`d(\cdot)` is a Grassmann distance metric and :math:`\mathbf{Y}` is a reference point on :math:`\mathcal{G}(p,n)`. :mod:`UQpy` offers two methods for solving this optimization, the GradientDescent and the stochastic Gradient descent methods, defined in 


The :class:`.karcher_mean` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.grassmann import karcher_mean

One can use the following command to instantiate the class :class:`.karcher_mean`

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.grassmann.karcher_mean
    :members:	
	
	
	

.. [1] T. Bendokat, R. Zimmermann, P.-A. Absil, A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects, 2020.

.. [2] P.-A. Absil, R. Mahony, and R. Sepulchre. Riemannian geometry of Grassmann manifolds with a view on algorithmic computation. Acta Applicandae Mathematica, 80(2):199{220, 2004.
