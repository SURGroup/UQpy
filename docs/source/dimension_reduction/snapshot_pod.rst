SnapshotPOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Snapshot Proper Orthogonal Decomposition (POD) method is the second variant of the POD method which considers the
decomposition of a dataset into deterministic temporal modes and random spatial coefficients. Essentially, this method
interchanges the time and position. In most problems the number of solution snapshots :math:`m` is less than the number
of dimensions :math:`n = N_x \times N_y` where :math:`N_x, N_y` are the grid dimensions. Thus, by using the
:class:`.SnapshotPOD` class, one can reconstruct solutions much faster (:cite:t:`POD_2`).

For the Snapshot POD, again a two-dimensional dataset :math:`\mathbf{U}\in \mathbb{R}^{n\times m}` is constructed where
:math:`m` is the number of snapshots and :math:`n` is the number of problem dimensions. The covariance matrix
:math:`\mathbf{C_s}`, is calculated as follows

.. math:: \mathbf{C_s} = \frac{1}{m-1} \mathbf{U} \mathbf{U}^T

The eigenvalue problem is solved and the temporal modes (eigenvectors) are calculated as

.. math:: \mathbf{C} A_s = \lambda A_s

Spatial coefficients are therefore calculated as :math:`\Phi_s = \mathbf{U}^T A_s`. Finally, a predefined number of
:math:`k`-POD temporal modes and spatial coefficients can be considered for the reconstruction of data as follows

.. math:: \mathbf{\tilde{u}}(\mathtt{x},t) = \sum_{i=1}^{k} A_{si}(t) \Phi_{si}(\mathtt{x})


SnapshotPOD Class
""""""""""""""""""""""""""""""

The :class:`.SnapshotPOD` class is imported using the following command:

>>> from UQpy.dimension_reduction.pod.SnapshotPOD import SnapshotPOD

One can use the following command to instantiate the class :class:`.SnapshotPOD`

Methods
^^^^^^^^^^
.. autoclass:: UQpy.dimension_reduction.pod.SnapshotPOD
    :members: run

Attributes
^^^^^^^^^^
.. autoattribute:: UQpy.dimension_reduction.pod.SnapshotPOD.reconstructed_solution
.. autoattribute:: UQpy.dimension_reduction.pod.SnapshotPOD.reduced_solution
.. autoattribute:: UQpy.dimension_reduction.pod.SnapshotPOD.U
.. autoattribute:: UQpy.dimension_reduction.pod.SnapshotPOD.eigenvalues
.. autoattribute:: UQpy.dimension_reduction.pod.SnapshotPOD.phi
