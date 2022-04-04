DirectPOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Direct Proper Orthogonal Decomposition (POD) is the first variant of the POD method and is used for the extraction
of a set of orthogonal spatial basis functions and corresponding time coefficients from a dataset. The
:class:`.DirectPOD` class is used for dimensionality reduction of datasets obtained by numerical simulations, given a desired level of accuracy.

Let us consider the solution of a numerical model of a differential equation :math:`\mathbf{u}(\mathtt{x},t)`, where
:math:`\mathtt{x} = (x,y,z)` is the position vector where the function is evaluated and :math:`t` is the time. The
idea behind the POD is to decompose the random vector field :math:`\mathbf{u}(\mathtt{x},t)`, into a set of
deterministic spatial functions :math:`\Phi_{k}{\mathtt{x}}`, multiplied by random time coefficients :math:`\alpha_{k}(t)`, so that:

.. math:: \mathbf{u}(\mathtt{x},t) =  \sum_{k=1}^{\infty}\alpha_{k}(t)\Phi_{k}{\mathtt{x}}

where :math:`\Phi_{k}{\mathtt{x}}` are the spatial POD modes and :math:`\alpha_{k}(t)` are the time coefficients.

The above decomposition is achieved by maximizing the energy that can be captured by the first :math:`n` spatial POD
modes :cite:t:`POD_1`. POD modes are orthonormal and thus one can write

.. math::  \iiint_{\mathtt{x}} \Phi_{k_{1}}{\mathtt{x}} \Phi_{k_{2}}{\mathtt{x}} d\mathtt{x} = \begin{cases}
    1, & \text{if $k_1 = k_2$}.\\
    0, & \text{if $k_1 \ne k_2$}
  \end{cases}

Furthermore, at each time coefficient :math:`\alpha_{k}(t)` only depends on the spatial mode :math:`\Phi_{k}{\mathtt{x}}`.
By multiplying the decomposition equation with :math:`\Phi_{k}{\mathtt{x}}` and integrating over space one obtains the following

.. math:: \alpha_{k}(t) = \iiint_{\mathtt{x}} \mathbf{u}(\mathtt{x},t) \Phi_{k}{\mathtt{x}} d\mathtt{x}

The POD method, often called Principal Component Analysis (PCA) in the field of statistics, is traditionally applied
to datasets obtained by numerical simulations for engineering problems (e.g. fluid mechanics, mechanics of materials,
aerodynamics) which produce finite-dimensional data containing the evolution of problem solutions in time.

For the Direct POD method, a two-dimensional dataset :math:`\mathbf{U}` is constructed where the :math:`m` is the
number of snapshots and :math:`n` is the number of problem dimensions. The covariance matrix is computed as follows

.. math:: \mathbf{C} = \frac{1}{m-1} \mathbf{U}^T \mathbf{U}

Next, the eigenvalue problem is solved for the covariance matrix as

.. math:: \mathbf{C} \Phi = \lambda \Phi

In total, :math:`n` eigenvalues :math:`\lambda_1,... \lambda_n` and a corresponding set of eigenvectors, arranged as
columns in an :math:`n \times n` matrix :math:`\Phi`. The :math:`n` columns of this matrix are the proper orthogonal
modes of the dataset. The original snapshot matrix :math:`\mathbf{U}`, can be expressed as the sum of the contributions
of the :math:`n` deterministic modes. The temporal coefficients are calculated as :math:`A = \mathbf{U} \Phi`. A
predefined number of :math:`k` POD spatial modes (eigenvectors) and temporal coefficients can be considered for the
reconstruction of data as follows

.. math:: \mathbf{\sim{u}}(\mathtt{x},t) =  \sum_{i=1}^{k}A(t)\Phi{\mathtt{x}}


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

