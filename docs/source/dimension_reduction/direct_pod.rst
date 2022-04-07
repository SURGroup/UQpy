DirectPOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Direct Proper Orthogonal Decomposition (POD) is the first variant of the POD method and is used for the extraction
of a set of orthogonal spatial basis functions and corresponding time coefficients from a dataset. The
:class:`.DirectPOD` class is used for dimensionality reduction of datasets obtained by numerical simulations, given a
desired level of accuracy.


For the Direct POD method, a two-dimensional dataset :math:`\mathbf{U}\in \mathbb{R}^{n\times m}` is constructed where
:math:`m` is the number of snapshots and :math:`n` is the number of problem dimensions. The covariance matrix is
computed as follows

.. math:: \mathbf{C} = \frac{1}{m-1} \mathbf{U}^T \mathbf{U}

Next, the eigenvalue problem is solved for the covariance matrix as

.. math:: \mathbf{C} \Phi = \lambda \Phi

In total, :math:`n` eigenvalues :math:`\lambda_1,... \lambda_n` and a corresponding set of eigenvectors, arranged as
columns in an :math:`n \times n` matrix :math:`\Phi`, are obtained. The :math:`n` columns of this matrix are the POD
modes of the dataset. The original snapshot matrix :math:`\mathbf{U}`, can be expressed as the sum of the contributions
of the :math:`n` deterministic modes. The temporal coefficients are calculated as :math:`A = \mathbf{U} \Phi`. A
predefined number of :math:`k` POD spatial modes (eigenvectors) and temporal coefficients can be considered for the
reconstruction of data as follows

.. math:: \mathbf{\tilde{u}}(\mathtt{x},t) =  \sum_{i=1}^{k}A_i(t)\Phi_i(\mathtt{x})


DirectPOD Class
""""""""""""""""""""""""""""""

The :class:`.DirectPOD` class is imported using the following command:

>>> from UQpy.dimension_reduction.pod.DirectPOD import DirectPOD

One can use the following command to instantiate the class :class:`.DirectPOD`

Methods
^^^^^^^^^^

.. autoclass:: UQpy.dimension_reduction.pod.DirectPOD
    :members: run

Attributes
^^^^^^^^^^
.. autoattribute:: UQpy.dimension_reduction.pod.DirectPOD.reconstructed_solution
.. autoattribute:: UQpy.dimension_reduction.pod.DirectPOD.reduced_solution

