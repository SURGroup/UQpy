.. _dimension_reduction_doc:

DimensionReduction
====================

.. automodule:: UQpy.DimensionReduction

This module contains the classes and methods to perform the point-wise and multi point data-based dimensionality reduction via projection onto the Grassmann manifold and Diffusion Maps, respectively. Further, interpolation in the tangent space centered at a given point on the Grassmann manifold can be performed. In addition, dataset reconstruction and dimension reduction can be performed via the Proper Orthogonal Decomposition method and the Higher-order Singular Value Decomposition for solution snapshots in the form of second-order tensors.

The module ``UQpy.DimensionReduction`` currently contains the following classes:

* ``Grassmann``: Class for for analysis of samples on the Grassmann manifold.

* ``DiffusionMaps``: Class for multi point data-based dimensionality reduction.

* ``POD``: Class for data reconstruction and data dimension reduction.


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

POD
--------------------------------
	
The ``POD`` class is the parent class of the ``DirectPOD``, ``SnapshotPOD`` and ``HOSVD`` classes that perform the Direct POD, Snapshot POD and Higher-order Singular Value Decomposition (HOSVD) respectively.

The Proper Orthogonal Decomposition (POD) is a post-processing technique which takes a given dataset and extracts a set of orthogonal basis functions and the corresponding coefficients. The idea of this method, is to analyze large amounts of data in order to gain a better understanding of the simulated processes and reduce noise. POD method has two variants, the Direct POD and Snapshot POD. In cases where the dataset is large, the Snapshot POD is recommended as it is much faster. 

The Higher-order Singular Value Decomposition (HOSVD) is the generalization of the matrix SVD, also called an orthogonal Tucker decomposition. HOSVD is used in cases where the solution snapshots are most naturally condensed into generalized matrices (tensors) and do not lend themselves naturally to vectorization. 

POD Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``POD`` class is imported using the following command:

>>> from DimensionReduction import POD

One can use the following command to instantiate the class ``POD``

.. autoclass:: UQpy.DimensionReduction.POD
    :members: 


DirectPOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
The Direct Proper Orthogonal Decomposition (POD) is the first variant of the POD method and is used for the extraction of a set of orthogonal spatial basis functions and corresponding time coefficients from a dataset. The ``DirectPOD`` class is used for dimensionality reduction of datasets obtained by numerical simulations, given a desired level of accuracy.

Let us consider the solution of a numerical model of a differential equation :math:`\mathbf{u}(\mathtt{x},t)`, where :math:`\mathtt{x} = (x,y,z)` is the position vector where the function is evaluated and :math:`t` is the time. The idea behind the POD is to decompose the random vector field :math:`\mathbf{u}(\mathtt{x},t)`, into a set of deterministic spatial functions :math:`\Phi_{k}{\mathtt{x}}`, multiplied by random time coefficients :math:`\alpha_{k}(t)`, so that:

.. math:: \mathbf{u}(\mathtt{x},t) =  \sum_{k=1}^{\infty}\alpha_{k}(t)\Phi_{k}{\mathtt{x}}

where :math:`\Phi_{k}{\mathtt{x}}` are the spatial POD modes and :math:`\alpha_{k}(t)` are the time coefficients.

The above decomposition is achieved by maximizing the energy that can be captured by the first :math:`n` spatial POD modes [9]_. POD modes are orthonormal and thus one can write

.. math::  \iiint_{\mathtt{x}} \Phi_{k_{1}}{\mathtt{x}} \Phi_{k_{2}}{\mathtt{x}} d\mathtt{x} = \begin{cases}
    1, & \text{if $k_1 = k_2$}.\\
    0, & \text{if $k_1 \ne k_2$}
  \end{cases}

Furthermore, at each time coefficient :math:`\alpha_{k}(t)` only depends on the spatial mode :math:`\Phi_{k}{\mathtt{x}}`. By multiplying the decomposition equation with :math:`\Phi_{k}{\mathtt{x}}` and integrating over space one obtains the following

.. math:: \alpha_{k}(t) = \iiint_{\mathtt{x}} \mathbf{u}(\mathtt{x},t) \Phi_{k}{\mathtt{x}} d\mathtt{x} 

The POD method, often called Principal Component Analysis (PCA) in the field of statistics, is traditionally applied to datasets obtained by numerical simulations for engineering problems (e.g. fluid mechanics, mechanics of materials, aerodynamics) which produce finite-dimensional data containing the evolution of problem solutions in time. 

For the Direct POD method, a two-dimensional dataset :math:`\mathbf{U}` is constructed where the :math:`m` is the number of snapshots and :math:`n` is the number of problem dimensions. The covariance matrix is computed as follows 

.. math:: \mathbf{C} = \frac{1}{m-1} \mathbf{U}^T \mathbf{U}

Next, the eigenvalue problem is solved for the covariance matrix as

.. math:: \mathbf{C} \Phi = \lambda \Phi

In total, :math:`n` eigenvalues :math:`\lambda_1,... \lambda_n` and a corresponding set of eigenvectors, arranged as columns in an :math:`n \times n` matrix :math:`\Phi`. The :math:`n` columns of this matrix are the proper orthogonal modes of the dataset. The original snapshot matrix :math:`\mathbf{U}`, can be expressed as the sum of the contributions of the :math:`n` deterministic modes. The temporal coefficients are calculated as :math:`A = \mathbf{U} \Phi`. A predefined number of :math:`k` POD spatial modes (eigenvectors) and temporal coefficients can be considered for the reconstruction of data as follows

.. math:: \mathbf{\sim{u}}(\mathtt{x},t) =  \sum_{i=1}^{k}A(t)\Phi{\mathtt{x}}


DirectPOD Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DirectPOD`` class is imported using the following command:

>>> from DimensionReduction import DirectPOD

One can use the following command to instantiate the class ``DirectPOD``

.. autoclass:: UQpy.DimensionReduction.DirectPOD
    :members: 
    
    
SnapshotPOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
The Snapshot Proper Orthogonal Decomposition (POD) method is the second variant of the POD method which considers the decomposition of a dataset into deterministic temporal modes and random spatial coefficients. Essentially, this method interchanges the time and position. In most problems the number of solution snapshots :math:`n` is less than the number of dimensions :math:`m = N_x \times N_y` where :math:`N_x, N_y` are the grid dimensions. Thus, by using the ``SnapshotPOD`` class one can reconstruct solutions much faster [10]_.

For the Snapshot POD the covariance matrix :math:`\mathbf{C_s}`, is calculated as follows 

.. math:: \mathbf{C_s} = \frac{1}{m-1} \mathbf{U} \mathbf{U}^T

The eigenvalue problem is solved and the temporal modes (eigenvectors) are calculated as

.. math:: \mathbf{C} A_s = \lambda A_s

Spatial coefficients are therefore calculated as :math:`\Phi_s = \mathbf{U}^T A_s`. Finally, a predefined number of :math:`k`-POD temporal modes and spatial coefficients can be considered for the reconstruction of data as follows

.. math:: \mathbf{\sim{u}}(\mathtt{x},t) = \sum_{i=1}^{k} A_s(t) \Phi_s \mathtt{x}


SnapshotPOD Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``SnapshotPOD`` class is imported using the following command:

>>> from DimensionReduction import SnapshotPOD

One can use the following command to instantiate the class ``SnapshotPOD``

.. autoclass:: UQpy.DimensionReduction.SnapshotPOD
    :members: 


HOSVD
--------------------------------
	
The Higher-order Singular Value Decomposition is a generalization of the classical SVD. Instead of vectorizing the solution snapshot into two-dimensional matrices we instead perform the dimension reduction directly to the generalized matrix, namely a  tensor. Let :math:`A \in \mathbb{R}^{I_1 \times I_2 \times ,..,\times I_N}` be an input Nth-order tensor containing the solution snapshots from a numerical simulation. The HOSVD decomposes :math:`A` as

.. math:: A = S \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)}...\times_N \mathbf{U}^{(N)}

where :math:`\times_N` denotes an n-mode tensor-matrix product.

By the above equation and the commutative property of n-mode product, one can obtain the
following relation

.. math:: S = A \times_1 {\mathbf{U}^{(1)}}^{T} ...\times_N {\mathbf{U}^{(N)}}^{T}

By using the properties of the n-mode product together with the definitions of Kronecker product, one can compute the n-mode unfolding of :math:`A` as

.. math:: A_{n} = \mathbf{U}^{(n)} S_{n} (\mathbf{U}^{(n+1)} \otimes \cdot\cdot\cdot \otimes \mathbf{U}^{(N)} \otimes \mathbf{U}^{(1)} \otimes \cdot\cdot\cdot \otimes \mathbf{U}^{(n-1)})^T

The ordering and orthogonality properties of :math:`S` imply that :math:`S(n)` has mutually orthogonal rows with Frobenius norms equal to :math:`\sigma_1^{n},\sigma_2^{n},...,\sigma_{I_n}^{n}`. Since the right and left resulting matrices in the above equation are both orthogonal the following can be defined

.. math:: \Sigma^{(n)} = \text{diag}(\sigma_1^{n},\sigma_2^{n},...,\sigma_{I_n}^{n})

Classical SVD must be performed to the unfolded matrices as 

.. math:: A = \mathbf{U}^{(n)} \Sigma^{(n)} {\mathbf{V}^{(n)}}^T

The HOSVD provides a set of bases :math:`\mathbf{U}^{(1)},...,\mathbf{U}^{(N-1)}` spanning each dimension of the snapshots plus a basis, :math:`\mathbf{U}^{(N)}`, spanning across the snapshots and the orthogonal core tensor, which generalizes the matrix of singular values. Finally, the reconstructed tensor can be computed as follows

.. math:: W(\xi_{k}) = \Sigma \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \cdot\cdot\cdot \times_N \mathbf{U}^{(N)}( \xi_{k}) 

where :math:`\mathbf{U}(N))( \xi_{k})` has dimension :math:`n \times 1`, where n is the number of snapshots and corresponds to the kth column of :math:`\mathbf{U}(N)` and is the number of independent bases that account for the desired accuracy of the reconstruction.

More information can be found in [11]_, [12]_.
 

HOSVD Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``HOSVD`` class is imported using the following command:

>>> from DimensionReduction import HOSVD

One can use the following command to instantiate the class ``HOSVD``

.. autoclass:: UQpy.DimensionReduction.HOSVD
    :members: 

|

.. [1] J. Maruskin, Introduction to Dynamical Systems and Geometric Mechanics, Solar Crest Publishing, LLC, 2012:p.165.

.. [2] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank represen6tation on Grassmann   manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.168.

.. [3] S. Sommer, T. Fletcher, X. Pennec, 1 - introduction to differential and Riemannian  geometry, in: X. Pennec, S. Sommer, T. Fletcher (Eds.), Riemannian Geometric Statistics in Medical Image Analysis, Academic Press, 2020, p.3–37.
	
.. [4] D. Giovanis, M. Shields, Uncertainty  quantification for complex systems with very high dimensional response using Grassmann manifold variations, Journal of Computational Physics, 2018, 364, p.393–415.

.. [5] B. Wang, Y. Hu, J. Gao, Y. Sun, B. Yin, Low rank representation on Grassmann manifolds: An extrinsic perspective, 2015, 167arXiv:1504.01807.

.. [6] M. T. Harandi, M. Salzmann, S. Jayasumana, R. Hartley, H. Li, Expanding the family of Grassmannian kernels: An embedding perspective, 2014, 1622014.arXiv:1407.1123.

.. [7] R. R. Coifman, S. Lafon. Diffusion maps. Applied Computational Harmonic Analysis, 2006, 21(1), p.5–30.

.. [8] R. R. Coifman, I. G. Kevrekidis, S. Lafon, M. Maggioni, and B. Nadler, Diffusionmaps, reduction coordinates, and low dimensional representation of stochastic systems, Multiscale Modeling and Simulation, 2008, 7(2), p.842–864.

.. [9] J. Weiss. A tutorial on the proper orthogonal decomposition. In: AIAA Aviation 2019 Forum. 2019. p. 3333.

.. [10] L. Sirovich. Turbulence and the dynamics of coherent structures. I. Coherent structures. Quarterly of applied mathematics, 1987, 45(3), 561-571.

.. [11] D. Giovanis, M. Shields. Variance‐based simplex stochastic collocation with model order reduction for high‐dimensional systems. International Journal for Numerical Methods in Engineering, 2019, 117(11), 1079-1116.

.. [12] L. De Lathauwer, B. De Moor, J. Vandewalle. A multilinear singular value decomposition. SIAM journal on Matrix Analysis and Applications, 2000, 21(4), 1253-1278.

.. toctree::
    :maxdepth: 2
