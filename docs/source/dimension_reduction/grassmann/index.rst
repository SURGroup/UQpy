Grassmann manifold
--------------------------------

In differential geometry the Grassmann manifold :math:`\mathcal{G}(p, n)` refers to a collection of
:math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space :cite:`Grassmann_1` :cite:`Grassmann_2`. A point on :math:`\mathcal{G}(p, n)` is typically represented as a :math:`n \times p` orthonormal matrix :math:`\mathbf{X}`, whose column spans the corresponding subspace.

.. toctree::
   :maxdepth: 1
   :caption: Methods


Exponential mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For point :math:`\mathbf{X}` we can define the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` as  the set of all matrices :math:`\mathbf{\Gamma}` at :math:`\mathbf{X}` such as

.. math:: \mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)} = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} : \mathbf{\Gamma}^T\mathbf{X}=\mathbf{0}\}


Consider two points :math:`\mathbf{X}` and :math:`\mathbf{Y}` on :math:`\mathcal{G}(p, n)` and the tangent space :math:`\mathcal{T}_{\mathbf{X}, \mathcal{G}(p,n)}` at :math:`\mathbf{X}`. The  geodesic, which refers to the shortest curve on the manifold connecting the two points, is defined as

.. math:: \mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T

.. math:: \Phi(t)=\mathrm{span}\left[\left(\mathbf{X}\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Phi}(0)=\mathbf{X}` and :math:`\mathbf{\Phi}(1)=\mathbf{Y}`.


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

    Optimization  <optimization>


Manifold Projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A collection of methods to project data on the Grassmann manifold.

The abstract :class:`.ManifoldProjection` class is the parent class that allows the user to define a set of methods that
must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.ManifoldProjection
    :members:

The :class:`.ManifoldProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.ManifoldProjection import ManifoldProjection

SVD Projection
~~~~~~~~~~~~~~~~~~~~~~

The :class:`.SvdProjection` class is used to project each data point of a given dataset onto a Grassmann manifold using the Singular Value Decomposition (SVD). The SVD factorizes a matrix :math:`\mathbf{X}\in \mathbb{R}^{n \times m}` into three matrices:

.. math:: \mathbf{X} =  \mathbf{U}  \mathbf{\Sigma}  \mathbf{V}^\intercal

where :math:`\mathbf{U}` and :math:`\mathbf{V}` are the matrices of left and right eigenvectors, respectively and :math:`\mathbf{\Sigma}` is a diagonal matrix containing the eigenvalues. Since :math:`\mathbf{U}` and :math:`\mathbf{V}` are orthonormal matrices we consider them to be representatives of the data point on the Grassmann manifold. The :class:`.SvdProjection` class allows the user to define the Grassmann manifold :math:`\mathcal{G}(p, n)` on which the data will reside by selecting the number of :math:`p-` planes, i.e., the rank of matrix :math:`\mathbf{U}` is equal to the number of :math:`p-` planes. It also provides the flexibility to define various compositions of Grassmann kernels using the :math:`\mathbf{U}` and :math:`\mathbf{V}` matrices.

The :class:`.SvdProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.SvdProjection import SvdProjection

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.SvdProjection
    :members:






Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:mod:`UQpy` offers the capability to interpolate points on the Grassmann :math:`\mathcal{G}(p,n)`. Consider we have a set of
:math:`n+1` points :math:`(t_0, \mathbf{X}_0), ..., (t_n, \mathbf{X}_n)`, with :math:`t_0 <...<t_n` and
:math:`\mathbf{X}_k \in \mathbb{R}^{p \times n}`,  and we want to find
a function :math:`p(x)` for which :math:`p(t_k)=\mathbf{X}_k` for :math:`k=0,..,n`. In this setting,
:math:`x` is a continuous independent variable and :math:`t_k` are called the nodes (or coordinates) of the interpolant.
However, since the Grassmann manifold has a nonlinear structure, interpolation can only be performed on the tangent space
which is a flat space. To this end, the steps required to interpolate a point on :math:`\mathcal{G}(p,n)` the  are the
following:

1. Calculate the Karcher mean of the given points on the manifold.
2. Project all points onto the tangent space with origin the Karcher mean.
3. Perform the interpolation on the tangent space using the available methods.
4. Map the interpolated point back onto the manifold.

The :class:`.ManifoldInterpolation` class provides a framework to perform these steps. To use this
class we need to import it first

>>> from UQpy.dimension_reduction.ManifoldInterpolation import ManifoldInterpolation

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.ManifoldInterpolation
    :members:


:py:mod:`UQpy` provides a collection of methods to perform the interpolation on the tangent space.

.. toctree::
   :hidden:
   :maxdepth: 1

    Methods <interpolation>


Examples
~~~~~~~~~~~~~~~~~~

.. toctree::

   Grassmann Manifold Examples <../auto_examples/dimension_reduction/grassmann/index>