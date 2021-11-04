Grassmann manifold
--------------------------------

In differential geometry the Grassmann manifold :math:`\mathcal{G}(p, n)` refers to a collection of
:math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space [1]_ [2]_. A point on :math:`\mathcal{G}(p, n)` is typically represented as a :math:`n \times p` orthonormal matrix :math:`\mathbf{X}`, whose column spans the corresponding subspace.  For point :math:`\mathbf{X}` we can define the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` as  the set of all matrices :math:`\mathbf{\Gamma}` at :math:`\mathbf{X}` such as 

.. math:: \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)} = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} : \mathbf{\Gamma}^T\mathbf{X}=\mathbf{0}\}


Consider two points :math:`\mathbf{X}` and :math:`\mathbf{Y}` on :math:`\mathcal{G}(p, n)` and the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` at :math:`\mathbf{X}`. The  geodesic, which refers to the shortest curve on the manifold connecting the two points, is defined as

.. math:: \mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T

.. math:: \Phi(t)=\mathrm{span}\left[\left(\mathbf{X}\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Phi}(0)=\mathbf{X}` and :math:`\mathbf{\Phi}(1)=\mathbf{Y}`.

The :class:`.Grassmann` class offers various methods to operate with data on the Grassmann manifold.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.Grassmann


.. toctree::
   :maxdepth: 1
   :caption: Methods

Exponential mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exponential map, denoted as :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}  \rightarrow \mathcal{G}(p, n)`, maps a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` to the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` in the direction :math:`\mathbf{\Gamma}`.


.. math::  \mathrm{exp}_{\mathbf{X}}(\mathbf{\Gamma})\equiv\mathbf{Y} 

.. math:: \mathbf{Y} = \mathrm{exp}_{\mathbf{X}}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{X}\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T


In order to use the method :meth:`.exp_map` one needs to import the :class:`.Grassmann` class from the :mod:`UQpy.dimension_reduction.grassmann_manifold` module

>>> from UQpy.dimension_reduction.grassmann_manifold import Grassmann
>>> Grassmann.exp_map()

Since :meth:`.exp_map` is a static method, it does not require instantiation of the class. 

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.Grassmann.exp_map




Logarithmic map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The logarithmic map, denoted as :math:`\mathcal{G}(p, n) \rightarrow  \mathcal{T}_{\mathbf{X}\mathcal{G}(p,n)}` maps the endpoint :math:`\mathbf{Y}=\Phi(1)` of the unique geodesic :math:`\Phi` that emanates from :math:`\mathbf{X}` to a tangent vector :math:`\mathbf{\Gamma} \in \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}`.


.. math:: \mathrm{log}_\mathbf{X}(\mathbf{Y})\equiv \mathbf{\Gamma} = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T 

In order to use the method :meth:`.log_map` one needs to import the :class:`.Grassmann` class from the :mod:`UQpy.dimension_reduction.grassmann_manifold` module

>>> from UQpy.dimension_reduction.grassmann_manifold import Grassmann
>>> Grassmann.log_map()

Since :meth:`.log_map` is a static method, it does not require instantiation of the class. 

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.Grassmann.log_map


Frechet variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Frechet variance is defined as the solution of the following minimization problem: 

.. math:: \sigma_{f}^2 = \mathrm{min}(\frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i - \mathbf{Y})^2)

where :math:`d(\cdot)` is a Grassmann distance metric and :math:`\mathbf{Y}` is a reference point on :math:`\mathcal{G}(p,n)`.

In order to use the method :meth:`.frechet_variance` one needs to import the :class:`.Grassmann` class from the :mod:`UQpy.dimension_reduction.grassmann_manifold` module

>>> from UQpy.dimension_reduction.grassmann_manifold import Grassmann
>>> Grassmann.frechet_variance()

Since :meth:`.frechet_variance` is a static method, it does not require instantiation of the class. 

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.Grassmann.frechet_variance
	

Karcher mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a set of points :math:`\{\mathbf{X}_i\}_{i=1}^N`  on :math:`\mathcal{G}(p,n)` the Karcher mean is defined as the solution of the following minimization problem: 

.. math:: \mu = \arg \mathrm{min}(\frac{1}{N}\sum_{i=1}^N d(\mathbf{X}_i - \mathbf{Y})^2)

where :math:`d(\cdot)` is a Grassmann distance metric and :math:`\mathbf{Y}` is a reference point on :math:`\mathcal{G}(p,n)`. 

In order to use the method :meth:`.karcher_mean` one needs to import the :class:`.Grassmann` class from the :mod:`UQpy.dimension_reduction.grassmann_manifold` module

>>> from UQpy.dimension_reduction.grassmann_manifold import Grassmann
>>> Grassmann.karcher_mean()

Since :meth:`.karcher_mean` is a static method, it does not require instantiation of the class. 

.. automethod:: UQpy.dimension_reduction.grassmann_manifold.Grassmann.karcher_mean
	
:mod:`UQpy` offers two classes for solving this optimization, the :class:`.GradientDescent` and the :class:`.StochasticGradientDescent`.

.. toctree::
   :maxdepth: 1
   :caption: Optimization methods	
   
  
  Optimization methods <optimization_methods/index>
  

Manifold projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A collection of methods to project data on the Grassmann manifold.

.. toctree::
   :hidden:
   :maxdepth: 1
   
  
  Methods <manifold_projections/index>	

.. [1] T. Bendokat, R. Zimmermann, P.-A. Absil, A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects, 2020.

.. [2] P.-A. Absil, R. Mahony, and R. Sepulchre. Riemannian geometry of Grassmann manifolds with a view on algorithmic computation. Acta Applicandae Mathematica, 80(2):199{220, 2004.
