Grassmann
--------------------------------
	
In differential geometry the Grassmann manifold :math:`\mathcal{G}_{n,p}` refers to a collection of
:math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space. The :class:`.Grassmann` class contains
methods to perform the projection of matrices onto the Grassmann manifold via singular value decomposition (SVD),
where their dimensionality are reduced. Further, a tangent space, where standard interpolation can be performed, is
constructed at a given reference point. Therefore, the mapping from the Grassmann manifold to the tangent space and
from the tangent space to the manifold are performed via the logarithmic and exponential mapping, respectively.
Moreover, additional quantities such as the Karcher mean, which correspond to the point on the Grassmann manifold
minimizing the squared distances to the other points on the same manifold. Further, the kernel defined on the Grassmann
manifold is implemented to estimate an affinity matrix to be used in kernel-based machine learning techniques.

A tangent space :math:`\mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n)`, which is a flat inner-product space, is defined as a
set of all tangent vectors at :math:`\mathcal{X}` [1]_, [2]_, [3]_; such as

.. math:: \mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n) = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} : \mathbf{\Gamma}^T\mathbf{\Psi}=\mathbf{0}\}

Where a point :math:`\mathcal{X} = \mathrm{span}\left(\mathbf{\Psi}\right) \in \mathcal{G}(p,n)` is invariant to the
choice of basis such that :math:`\mathrm{span}\left(\mathbf{\Psi}\right) = \mathrm{span}\left(\mathbf{R\Psi}\right)`,
with :math:`\mathbf{R} \in SO(p)`, where :math:`SO(p)` is the special orthogonal group.

One can write the exponential map (from the tangent space to the manifold) locally as ([4]_, [5]_)

.. math:: \mathrm{exp}_{\mathcal{X}_0}(\mathbf{\Gamma}) = \mathbf{\Psi}_1

Denoting :math:`\mathbf{\Gamma}` by its singular value decomposition
:math:`\mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T` one can write a point on the Grassmann manifold
:math:`\mathbf{\Psi}_1`, considering a reference point :math:`\mathbf{\Psi}_0`, as

.. math:: \mathbf{\Psi}_1 = \mathrm{exp}_{\mathcal{X}_0}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{\Psi}_0\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T


Equivalently, the logarithmic map :math:`\mathrm{log}_\mathcal{X}:\mathcal{G}(p,n) \rightarrow \mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n)`
is defined locally as

.. math:: \mathrm{log}_\mathcal{X}(\mathbf{\Psi}_1) = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T

One can write the geodesic as

.. math:: \gamma(t)=\mathrm{span}\left[\left(\mathbf{\Psi}_0\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Psi}_0`, if :math:`t=0`, and :math:`\mathbf{\Psi}_1`, :math:`t=1`.



Grassmann Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Grassmann` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassman.Grassman import Grassmann

One can use the following command to instantiate the class :class:`.Grassmann`

.. autoclass:: UQpy.dimension_reduction.grassman.Grassman
    :members:


.. [1] J. Maruskin, Introduction to Dynamical Systems and Geometric Mechanics, Solar Crest Publishing, LLC, 2012:p.165.

.. [2] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank represen6tation on Grassmann   manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.168.

.. [3] S. Sommer, T. Fletcher, X. Pennec, 1 - introduction to differential and Riemannian  geometry, in: X. Pennec, S. Sommer, T. Fletcher (Eds.), Riemannian Geometric Statistics in Medical Image Analysis, Academic Press, 2020, p.3–37.

.. [4] D. Giovanis, M. Shields, Uncertainty  quantification for complex systems with very high dimensional response using Grassmann manifold variations, Journal of Computational Physics, 2018, 364, p.393–415.

.. [5] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank representation on Grassmann manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.

.. [6] M. T. Harandi, M. Salzmann, S. Jayasumana, R. Hartley, H. Li, Expanding the family of Grassmannian kernels: An embedding perspective, 2014, 1622014.arXiv:1407.1123.
