Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:mod:`UQpy` offers the capability to interpolate points on the Grassmann :math:`\mathcal{G}(p,n)`. Consider we have a
set of :math:`n+1` points :math:`(t_0, \mathbf{X}_0), ..., (t_n, \mathbf{X}_n)`, with :math:`t_0 <...<t_n` and
:math:`\mathbf{X}_k \in \mathbb{R}^{p \times n}`,  and we want to find
a function :math:`p(x)` for which :math:`p(t_k)=\mathbf{X}_k` for :math:`k=0,..,n` where
:math:`x` is a continuous independent variable and :math:`t_k` are called the nodes (or coordinates) of the interpolant.
However, since the Grassmann manifold has a nonlinear structure, interpolation can only be performed on the tangent
space, which is a flat inner-product space. Therefore the following steps are required to interpolate on
:math:`\mathcal{G}(p,n)`:

1. Calculate the Karcher mean of the given points on the manifold.
2. Project all points onto the tangent space with origin at the Karcher mean.
3. Perform the interpolation on the tangent space using the available methods.
4. Map the interpolated point back onto the manifold.

The :class:`.GrassmannInterpolation` class provides a framework to perform these steps. To use this
class we need to import it as follows:

>>> from UQpy.dimension_reduction.grassmann_manifold.GrassmannInterpolation import GrassmannInterpolation

We must then instantiate a :class:`.GrassmannInterpolation` object and then invoke the :py:meth:`.interpolate_manifold`
method as:

>>> X = GrassmannInterpolation()
>>> Y = X.interpolate_manifold(point)


Methods
~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.dimension_reduction.GrassmannInterpolation
    :members:
