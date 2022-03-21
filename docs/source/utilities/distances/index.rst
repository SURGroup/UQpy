Distances
--------------------------------------

A collection of distances between points in the Euclidean space and between subspaces on the Grassmann manifold.

The Grassmannian (or Riemannian) distance is a metric that assigns nonnegative values to each pair of subspaces
:math:`Y_1, Y_2 \in \mathbb{R}^{n \times p}` on the Grassmann manifold :math:`\mathcal{G}(p, n)`. Formally,
is defined as the length of the shortest geodesic connecting the two points on :math:`\mathcal{G}(p, n)`.
:py:mod:`UQpy` introduces various Grassmann distances derived from the principal angles :cite:`Distances_1`.

.. math:: 0 \leq \theta_1 \leq \theta_2 \leq \ldots \leq \theta_p \leq \pi/2,

where the first principal angle :math:`\theta_1` is the smallest angle between all pairs of unit vectors in the
first and the second subspaces. Practically, the principal angles can be calculated from the singular value
decomposition (SVD) of :math:`\mathbf{Y}_1'\mathbf{Y}_2`,

.. math:: \mathbf{Y}_1'\mathbf{Y}_2 = \mathbf{U}\cos(\Theta)\mathbf{V}'

where :math:`\cos(\Theta)=\text{diag}(\cos\theta_1, \ldots,\cos\theta_p)`. This definition of distance can be extended
to cases where :math:`\mathbf{Y}_1` and :math:`\mathbf{Y}_2` have different number of columns :math:`p`. More
information can be found in :cite:`Distances_2`.

Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Distances` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.Distance import Distance

.. autoclass:: UQpy.utilities.distances.baseclass.Distance
    :members: compute_distance


.. toctree::
   :maxdepth: 1

    Grassmannian Distances <grassmann_distances>
    Euclidean Distances <euclidean_distances>
