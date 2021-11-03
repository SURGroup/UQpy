Higher-order Singular Value Decomposition
----------------------------------------------

The Higher-order Singular Value Decomposition (HOSVD) is the generalization of the matrix SVD, also called an orthogonal
Tucker decomposition. HOSVD is used in cases where the solution snapshots are most naturally condensed into generalized
matrices (tensors) and do not lend themselves naturally to vectorization. Let :math:`A \in \mathbb{R}^{I_1 \times I_2 \times ,..,\times I_N}` be an input Nth-order
tensor containing the solution snapshots from a numerical simulation. The HOSVD decomposes :math:`A` as

.. math:: A = S \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)}...\times_N \mathbf{U}^{(N)}

where :math:`\times_N` denotes an n-mode tensor-matrix product.

By the above equation and the commutative property of n-mode product, one can obtain the
following relation

.. math:: S = A \times_1 {\mathbf{U}^{(1)}}^{T} ...\times_N {\mathbf{U}^{(N)}}^{T}

By using the properties of the n-mode product together with the definitions of Kronecker product, one can compute the
n-mode unfolding of :math:`A` as

.. math:: A_{n} = \mathbf{U}^{(n)} S_{n} (\mathbf{U}^{(n+1)} \otimes \cdot\cdot\cdot \otimes \mathbf{U}^{(N)} \otimes \mathbf{U}^{(1)} \otimes \cdot\cdot\cdot \otimes \mathbf{U}^{(n-1)})^T

The ordering and orthogonality properties of :math:`S` imply that :math:`S(n)` has mutually orthogonal rows with
Frobenius norms equal to :math:`\sigma_1^{n},\sigma_2^{n},...,\sigma_{I_n}^{n}`. Since the right and left resulting
matrices in the above equation are both orthogonal the following can be defined

.. math:: \Sigma^{(n)} = \text{diag}(\sigma_1^{n},\sigma_2^{n},...,\sigma_{I_n}^{n})

Classical SVD must be performed to the unfolded matrices as

.. math:: A = \mathbf{U}^{(n)} \Sigma^{(n)} {\mathbf{V}^{(n)}}^T

The HOSVD provides a set of bases :math:`\mathbf{U}^{(1)},...,\mathbf{U}^{(N-1)}` spanning each dimension of the
snapshots plus a basis, :math:`\mathbf{U}^{(N)}`, spanning across the snapshots and the orthogonal core tensor, which
generalizes the matrix of singular values. Finally, the reconstructed tensor can be computed as follows

.. math:: W(\xi_{k}) = \Sigma \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \cdot\cdot\cdot \times_N \mathbf{U}^{(N)}( \xi_{k})

where :math:`\mathbf{U}(N))( \xi_{k})` has dimension :math:`n \times 1`, where n is the number of snapshots and
corresponds to the kth column of :math:`\mathbf{U}(N)` and is the number of independent bases that account for the desired accuracy of the reconstruction.

More information can be found in [11]_, [12]_.


HOSVD Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.HigherOrderSVD` class is imported using the following command:

>>> from UQpy.dimension_reduction.hosvd.HigherOrderSVD import HigherOrderSVD

One can use the following command to instantiate the class :class:`.HigherOrderSVD`

.. autoclass:: UQpy.dimension_reduction.hosvd.HigherOrderSVD
    :members:


.. [11] D. Giovanis, M. Shields. Variance‐based simplex stochastic collocation with model order reduction for high‐dimensional systems. International Journal for Numerical Methods in Engineering, 2019, 117(11), 1079-1116.

.. [12] L. De Lathauwer, B. De Moor, J. Vandewalle. A multilinear singular value decomposition. SIAM journal on Matrix Analysis and Applications, 2000, 21(4), 1253-1278.
