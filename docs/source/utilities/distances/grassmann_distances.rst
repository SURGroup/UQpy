Grassmann Distances
--------------------------------------

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

GrassmannianDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GrassmannianDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.GrassmannianDistance import GrassmannianDistance

The abstract :class:`.GrassmannianDistance` class is a blueprint for all classes in :mod:`.grassmannian_distances` module.
It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.utilities.distances.baseclass.GrassmannianDistance
    :members: calculate_distance_matrix



Asimov
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.AsimovDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.AsimovDistance import AsimovDistance

One can use the following command to instantiate the class :class:`.AsimovDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.AsimovDistance
    :members:


Binet-Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BinetCauchyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance import BinetCauchyDistance

One can use the following command to instantiate the class :class:`.BinetCauchyDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance
    :members:


Fubini-Study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.FubiniStudyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance import FubiniStudyDistance

One can use the following command to instantiate the class :class:`.FubiniStudyDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance
    :members:



Geodesic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GeodesicDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.GeodesicDistance import GeodesicDistance

One can use the following command to instantiate the class :class:`.GeodesicDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.GeodesicDistance
    :members:


Martin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MartinDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.MartinDistance import MartinDistance

One can use the following command to instantiate the class :class:`.MartinDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.MartinDistance
    :members:



Procrustes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProcrustesDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance import ProcrustesDistance

One can use the following command to instantiate the class :class:`.ProcrustesDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance
    :members:


Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProjectionDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProjectionDistance import ProjectionDistance

One can use the following command to instantiate the class :class:`.ProjectionDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProjectionDistance
    :members:


Spectral
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SpectralDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.SpectralDistance import SpectralDistance

One can use the following command to instantiate the class :class:`.SpectralDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.SpectralDistance
    :members:
