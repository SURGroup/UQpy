DiffusionMaps
--------------------------------

In nonlinear dimensionality reduction Diffusion Maps corresponds to a technique used to reveal the intrinsic structure
of data sets based on a diffusion process over the data. In particular, the eigenfunctions of Markov matrices defining a
random walk on the data are used to obtain a coordinate system represented by the diffusion coordinates revealing the
embedded geometry of the data. Moreover, the diffusion coordinates are defined on a Euclidean space where usual metrics
define the distances between pairs of data points. Thus, the diffusion maps create a connection between the spectral
properties of the diffusion process and the intrinsic geometry of the data resulting in a multiscale representation of
the data.

To present this method let's assume measure space :math:`(X, \mathcal{A}, \mu)`, where :math:`X` is the dataset,
:math:`\mathcal{A}` is a :math:`\sigma-`algebra on the set :math:`X`, and :math:`\mu` a measure; and a non-negative
symmetric kernel :math:`k: X \times X \rightarrow \mathbb{R}` representing the pairwise affinity of the data points in a
symmetric graph; one can define the connectivity between two points as the transition probability in a random walk using
the kernel :math:`k`. Therefore, the diffusion maps technique can be based on a normalized graph Laplacian construction [7]_, [8]_, where

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

The :class:`.DiffusionMaps` class is imported using the following command:

>>> from UQpy.dimension_reduction.DiffusionMaps import DiffusionMaps

One can use the following command to instantiate the class :class:`.DiffusionMaps`

.. autoclass:: UQpy.dimension_reduction.DiffusionMaps
    :members:

.. [7] R. R. Coifman, S. Lafon. Diffusion maps. Applied Computational Harmonic Analysis, 2006, 21(1), p.5–30.

.. [8] R. R. Coifman, I. G. Kevrekidis, S. Lafon, M. Maggioni, and B. Nadler, Diffusionmaps, reduction coordinates, and low dimensional representation of stochastic systems, Multiscale Modeling and Simulation, 2008, 7(2), p.842–864.