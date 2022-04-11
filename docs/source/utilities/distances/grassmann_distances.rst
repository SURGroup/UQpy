Grassmann Distances
--------------------------------------

The Grassmannian (or Riemannian) distance assigns a nonnegative value to measure distance between a pair of subspaces
:math:`\mathbf{X}_0, \mathbf{X}_1 \in \mathbb{R}^{n \times p}` on the Grassmann manifold :math:`\mathcal{G}(p, n)`.
Formally, a Grassmann distance measures the length of the geodesic path connecting the two points on
:math:`\mathcal{G}(p, n)` where all distances are a function of the principal angles between subspaces. :py:mod:`UQpy`
introduces various Grassmann distances derived from the principal angles :cite:`Distances_1`.

.. math:: 0 \leq \theta_1 \leq \theta_2 \leq \ldots \leq \theta_p \leq \pi/2,

Practically, the principal angles can be calculated from the singular value decomposition (SVD) of
:math:`\mathbf{X}_0^T\mathbf{X}_1`, as

.. math:: \mathbf{X}_0^T\mathbf{X}_1 = \mathbf{U}\cos(\Theta)\mathbf{V}'

where :math:`\cos(\Theta)=\text{diag}(\cos\theta_1, \ldots,\cos\theta_p)`. This definition of distance can be extended
to cases where :math:`\mathbf{X}_0` and :math:`\mathbf{X}_1` have different number of columns :math:`p`. More
information can be found in :cite:`Distances_2`.

GrassmannianDistance Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The abstract :class:`UQpy.utilities.distances.baseclass.GrassmannianDistance` class is the base class for all
Grassmannian distances in :py:mod:`UQpy`. It provides a blueprint for classes in the :mod:`.grassmannian_distances`
module and allows the user to define a set of methods that must be created within any child classes built
from this abstract class.

.. autoclass:: UQpy.utilities.distances.baseclass.GrassmannianDistance
    :members: calculate_distance_matrix

The :class:`.GrassmannianDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.GrassmannianDistance import GrassmannianDistance

List of Available Distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the distances classes below are subclasses of the :class:`.GrassmannianDistance` class. New distances can be written
as subclasses having a :py:meth:`compute_distance` method.

Asimov Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Asimov distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0` and
:math:`\mathbf{X}_1`, is given by:

.. math:: d_A(\mathbf{X}_0,\mathbf{X}_1) = \cos^{-1}||\mathbf{X}_0^T\mathbf{X}_1||_2 = \max(\Theta)

The :class:`.AsimovDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.AsimovDistance import AsimovDistance

One can use the following command to instantiate the :class:`.AsimovDistance` class.


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.AsimovDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.AsimovDistance.distance_matrix

Binet-Cauchy Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Binet-Cauchy distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0` and
:math:`\mathbf{X}_1`, is given by:

.. math:: d_{BC}(\mathbf{X}_0,\mathbf{X}_1) = \left[1-(\det\mathbf{X}_0^T\mathbf{X}_1)^2\right]^{1/2} = \left[1-\prod_{l}\cos^2(\Theta_l)\right]^{1/2}


The :class:`.BinetCauchyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance import BinetCauchyDistance

One can use the following command to instantiate the :class:`.BinetCauchyDistance` class.


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance.distance_matrix


Fubini-Study Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fubini-Study distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0` and
:math:`\mathbf{X}_1`, is given by:

.. math:: d_{FS}(\mathbf{X}_0,\mathbf{X}_1) = \cos^{-1}\left(\prod_{l}\cos(\Theta_l)\right)

The :class:`.FubiniStudyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance import FubiniStudyDistance

One can use the following command to instantiate the :class:`.FubiniStudyDistance` class.


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance.distance_matrix


Geodesic Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Geodesic (or Arc-length) distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0`
and :math:`\mathbf{X}_1`, is given by:

.. math:: d_{G}(\mathbf{X}_0,\mathbf{X}_1) = ||\boldsymbol\Theta ||_2 = \left(\sum \Theta^2_l\right)^{1/2}

The :class:`.GeodesicDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.GeodesicDistance import GeodesicDistance

One can use the following command to instantiate the class :class:`.GeodesicDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.GeodesicDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.GeodesicDistance.distance_matrix

Martin Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Martin distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0`
and :math:`\mathbf{X}_1`, is given by:

.. math:: d_{M}(\mathbf{X}_0,\mathbf{X}_1) = \left[\log\prod_{l}1/\cos^2(\Theta_l)\right]^{1/2}

The :class:`.MartinDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.MartinDistance import MartinDistance

One can use the following command to instantiate the :class:`.MartinDistance` class.


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.MartinDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.MartinDistance.distance_matrix



Procrustes Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Procrustes distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0`
and :math:`\mathbf{X}_1`, is given by:

.. math:: d_{P}(\mathbf{X}_0,\mathbf{X}_1) = ||\mathbf{X}_0\mathbf{U}-\mathbf{X}_1\mathbf{V} ||_F = 2\left[\sum_{l}\sin^2(\Theta_l/2)\right]^{1/2}

The :class:`.ProcrustesDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance import ProcrustesDistance

One can use the following command to instantiate the :class:`.ProcrustesDistance` class.

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance.distance_matrix

Projection Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Projection distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0`
and :math:`\mathbf{X}_1`, is given by:

.. math:: d_{Pr}(\mathbf{X}_0,\mathbf{X}_1) = ||\mathbf{X}_0\mathbf{X}_0^T-\mathbf{X}_1\mathbf{X}_1^T ||_2 =  \left(\sum_{l} \sin^2(\Theta_l)\right)^{1/2}

The :class:`.ProjectionDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProjectionDistance import ProjectionDistance

One can use the following command to instantiate the :class:`.ProjectionDistance` class.

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProjectionDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.ProjectionDistance.distance_matrix


Spectral Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Spectral distance between two subspaces defined by the orthonormal matrices, :math:`\mathbf{X}_0`
and :math:`\mathbf{X}_1`, is given by:

.. math:: d_{S}(\mathbf{X}_0,\mathbf{X}_1) = ||\mathbf{X}_0\mathbf{U}-\mathbf{X}_1\mathbf{V} ||_2 = 2\sin( \max(\Theta_l)/2)


The :class:`.SpectralDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.SpectralDistance import SpectralDistance

One can use the following command to instantiate the :class:`.SpectralDistance` class.

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.SpectralDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.grassmannian_distances.SpectralDistance.distance_matrix
