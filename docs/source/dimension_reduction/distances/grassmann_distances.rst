Grassmann Distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Grassmannian (or Riemannian) distance is a metric that assigns nonnegative values to each pair of subspaces :math:`Y_1, Y_2 \in \mathbb{R}^{n \times p}` on the Grassmann manifold :math:`\mathcal{G}(p, n)`. Formally, is defined as the length of the shortest geodesic connecting the two points on :math:`\mathcal{G}(p, n)`. :py:mod:`UQpy` introduces various Grassmann distances derived from the principal angles [1]_

.. math:: 0 \leq \theta_1 \leq \theta_2 \leq \ldots \leq \theta_p \leq \pi/2,

where the first principal angle :math:`\theta_1` is the smallest angle between all pairs of unit vectors in the first and the second subspaces. Practically, the principal angles can be calculated from the singular value decomposition (SVD) of :math:`Y_1'Y_2`,

.. math:: Y_1'Y_2 = U\cos(\Theta)V'

where :math:`\cos(\Theta)=\text{diag}(\cos\theta_1, \ldots,\cos\theta_p)`. This definition of distance can be extended to cases where :math:`Y_1` and :math:`Y_2` have different number of columns :math:`p`. More information can be found in [2]_.


Asimov Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.AsimovDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.grassmann.AsimovDistance import AsimovDistance

One can use the following command to instantiate the class :class:`.AsimovDistance`

.. autoclass:: UQpy.dimension_reduction.distances.grassmann.AsimovDistance
    :members:





.. [1] G. H. Golub and C.F.V. Loan. Matrix computations (3rd ed.). Baltimore, MD, USA: Johns Hopkins University Press, 1996.

.. [2]  K. Ye and L.H. Lim, Schubert varieties and distances between subspaces of different dimensions, SIAM Journal on Matrix Analysis and Applications, 2016, 37, 1176–1197.
