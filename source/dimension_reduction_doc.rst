.. _dimension_reduction_doc:

DimensionReduction
====================

.. automodule:: UQpy.DimensionReduction

This module contains the classes and methods to perform the point-wise and multi point data-based dimensionality reduction via projection onto the Grassmann manifold and Diffusion Maps, respectively. Further, interpolation in the tangent space centered at a given point on the Grassmann manifold can be performed.

The module ``UQpy.DimensionReduction`` currently contains the following classes:

* ``Grassmann``: Class for for analysis of samples on the Grassmann manifold.

* ``DiffusionMaps``: Class for multi point data-based dimensionality reduction.


Grassmann
--------------------------------
	
In differential geometry the Grassmann manifold :math:`\mathcal{G}_{n,p}` refers to a collection of :math:`p`-dimensional subspaces embedded in a :math:`n`-dimensional vector space. The ``Grassmann`` class contains methods to perform the projection of matrices onto the Grassmann manifold via singular value decomposition (SVD), where their dimensionality are reduced. Further, a tangent space, where standard interpolation can be performed, is constructed at a given reference point. Therefore, the mapping from the Grassmann manifold to the tangent space and from the tangent space to the manifold are performed via the logarithmic and exponential mapping, respectively. Moreover, additional quantities such as the Karcher mean, which correspond to the point on the Grassmann manifold minimizing the squared distances to the other points on the same manifold. Further, the kernel defined on the Grassmann manifold is implemented to estimate an affinity matrix to be used in kernel-based machine learning techniques.

A tangent space :math:`\mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n)`, which is a flat inner-product space, is defined as a set of all tangent vectors at :math:`\mathcal{X}` [1]_, [2]_, [3]_; such as 

.. math:: \mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n) = \{\mathbf{\Gamma} \in \mathbb{R}^{n \times p} : \mathbf{\Gamma}^T\mathbf{\Psi}=\mathbf{0}\}

Where a point :math:`\mathcal{X} = \mathrm{span}\left(\mathbf{\Psi}\right) \in \mathcal{G}(p,n)` is invariant to the choice of basis such that :math:`\mathrm{span}\left(\mathbf{\Psi}\right) = \mathrm{span}\left(\mathbf{R\Psi}\right)`, with :math:`\mathbf{R} \in SO(p)`, where :math:`SO(p)` is the special orthogonal group.

One can write the exponential map (from the tangent space to the manifold) locally as ([4]_, [5]_)

.. math:: \mathrm{exp}_{\mathcal{X}_0}(\mathbf{\Gamma}) = \mathbf{\Psi}_1

Denoting :math:`\mathbf{\Gamma}` by its singular value decomposition :math:`\mathbf{\Gamma} = \mathbf{U}\mathbf{S}\mathbf{V}^T` one can write a point on the Grassmann manifold :math:`\mathbf{\Psi}_1`, considering a reference point :math:`\mathbf{\Psi}_0`, as

.. math:: \mathbf{\Psi}_1 = \mathrm{exp}_{\mathcal{X}_0}(\mathbf{U}\mathbf{S}\mathbf{V}^T) = \mathbf{\Psi}_0\mathbf{V}\mathrm{cos}\left(\mathbf{S}\right)\mathbf{Q}^T+\mathbf{U}\mathrm{sin}\left(\mathbf{S}\right)\mathbf{Q}^T


Equivalently, the logarithmic map :math:`\mathrm{log}_\mathcal{X}:\mathcal{G}(p,n) \rightarrow \mathcal{T}_{\mathcal{X}}\mathcal{G}(p,n)` is defined locally as

.. math:: \mathrm{log}_\mathcal{X}(\mathbf{\Psi}_1) = \mathbf{U}\mathrm{tan}^{-1}\left(\mathbf{S}\right)\mathbf{V}^T

One can write the geodesic as

.. math:: \gamma(t)=\mathrm{span}\left[\left(\mathbf{\Psi}_0\mathbf{V}\mathrm{cos}(t\mathbf{S})+\mathbf{U}\mathrm{sin}(t\mathbf{S})\right)\mathbf{V}^T\right]

where :math:`\mathbf{\Psi}_0`, if :math:`t=0`, and :math:`\mathbf{\Psi}_1`, :math:`t=1`.

The geodesic distance :math:`d_{\mathcal{G}(p,n)}\left(\mathbf{\Psi}_0,\mathbf{\Psi}_1\right)` between two points on $\mathcal{G}(p,n)$ corresponds to the distance over the geodesic :math:`\gamma(t)` and it is given by

.. math:: d_{\mathcal{G}(p,n)}\left(\mathbf{\Psi}_0,\mathbf{\Psi}_1\right) = ||\mathbf{\Theta}||_2

where :math:`\mathbf{\Theta} = \left(\theta_1, \theta_2, \dots, \theta_p \right)` contains the principal angles. Several definitions of distance on :math:`\mathcal{G}(p,n)` can be found in the literature.

In several applications the use of subspaces is essential to describe the underlying geometry of data. However, it is well-known that subspaces do not follow the Euclidean geometry because they lie on the Grassmann manifold. Therefore, working with subspaces requires the definition of an embedding structure of the Grassmann manifold into a Hilbert space. Thus, using positive definite kernels is studied as a solution to this problem. In this regard, a real-valued positive definite kernel is defined as a symmetric function :math:`k:\mathcal{X}\times \mathcal{X} \rightarrow \mathbb{R}` if and only if :math:`\sum^n_{I,j=1}c_i c_j k(x_i,x_j) \leq 0` for :math:`n \in \mathbb{N}`, :math:`x_i in \mathcal{X}` and :math:`c_i \in \mathbb{R}`. Further, the Grassmann kernel can be defined as a function :math:`k:\mathcal{G}(p,n)\times \mathcal{G}(p,n) \rightarrow \mathbb{R}` if it is well-defined and positive definite [6]_.

Grassmann Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Grassmann`` class is imported using the following command:

>>> from DimensionReduction import Grassmann

One can use the following command to instantiate the class ``Grassmann``

.. autoclass:: UQpy.DimensionReduction.Grassmann
    :members:  

DiffusionMaps
--------------------------------

In nonlinear dimensionality reduction Diffusion Maps corresponds to a technique used to reveal the intrinsic structure of data sets based on a diffusion process over the data. In particular, the eigenfunctions of Markov matrices defining a random walk on the data are used to obtain a coordinate system represented by the diffusion coordinates revealing the embedded geometry of the data. Moreover, the diffusion coordinates are defined on a Euclidean space where usual metrics define the distances between pairs of data points. Thus, the diffusion maps create a connection between the spectral properties of the diffusion process and the intrinsic geometry of the data resulting in a multiscale representation of the data.

To present this method let's assume measure space :math:`(X, \mathcal{A}, \mu)`, where :math:`X` is the dataset, :math:`\mathcal{A}` is a :math:`\sigma-`algebra on the set :math:`X`, and :math:`\mu` a measure; and a non-negative symmetric kernel :math:`k: X \times X \rightarrow \mathbb{R}` representing the pairwise affinity of the data points in a symmetric graph; one can define the connectivity between two points as the transition probability in a random walk using the kernel :math:`k`. Therefore, the diffusion maps technique can be based on a normalized graph Laplacian construction [7]_, [8]_, where

.. math:: p(x, y) = \frac{k(x,y)}{\int_X k(x,y)d\mu(y)}

with 

.. math:: \int_X p(x,y)d\mu(y) = 1

can be viewed as the one-step transition probability . Therefore, to construct the transition probability one can resort to the graph Laplacian normalization. In this regard, one can consider that :math:`L_{i,j} = k(x_i,x_j)` must be normalized such that :math:`\tilde{L}_{i,j} = D^{-\alpha}LD^{-\alpha}`, where

.. math: D_{i,i} = \sum_j L_{i,j}

is a diagonal matrix. Next, a new matrix $D$ is obtained from :math:`\tilde{L}`, thus

.. math:: \tilde{D}_{i,i} = \sum_j \tilde{L}_{i,j}

Therefore, the transition probability :math:`M_{i,j} = p(x_j,t|x_i)` can be obtained after the graph Laplacian normalization of :math:`\tilde{L}` such as

.. math:: M = \tilde{D}^{-1}\tilde{L}

From the eigendecomposition of :math:`M`, one can obtain the eigenvectors :math:`(\psi_0, \psi_1, \dots, \psi_N)` and their respective eigenvalues :math:`(\lambda_0, \lambda_1, \dots, \lambda_N)`. However, only :math:`k` eigenvectors and eigenvalues suffice. Thus, the diffusion coordinates are given by :math:`\Psi_i = \lambda_i \psi_i` with :math:`i=1,\dots,k`. 


Diffusion Maps Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DiffusionMaps`` class is imported using the following command:

>>> from DimensionReduction import DiffusionMaps

One can use the following command to instantiate the class ``DiffusionMaps``

.. autoclass:: UQpy.DimensionReduction.DiffusionMaps
    :members:  

.. [1] J. Maruskin, Introduction to Dynamical Systems and Geometric Mechanics, Solar Crest Publishing, LLC, 2012:p.165.

.. [2] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank represen6tation on Grassmann   manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.168.

.. [3] S. Sommer, T. Fletcher, X. Pennec, 1 - introduction to differential and Riemannian  geometry, in: X. Pennec, S. Sommer, T. Fletcher (Eds.), Riemannian Geometric Statistics in Medical Image Analysis, Academic Press, 2020, p.3–37.
	
.. [4] D. Giovanis, M. Shields, Uncertainty  quantification for complex systems with very high dimensional response using Grassmann manifold variations, Journal of Computational Physics, 2018, 364, p.393–415.

.. [5] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank representation on Grassmann manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.

.. [6] M. T. Harandi, M. Salzmann, S. Jayasumana, R. Hartley, H. Li, Expanding the family of Grassmannian kernels: An embedding perspective, 2014, 1622014.arXiv:1407.1123.

.. [7] R. R. Coifman, S. Lafon. Diffusion maps. Applied Computational Harmonic Analysis, 2006, 21(1), p.5–30.

.. [8] R. R. Coifman, I. G. Kevrekidis, S. Lafon, M. Maggioni, and B. Nadler, Diffusionmaps, reduction coordinates, and low dimensional representation of stochastic systems, Multiscale Modeling and Simulation, 2008, 7(2), p.842–864.

.. toctree::
    :maxdepth: 2
