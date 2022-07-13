Proper Orthogonal Decomposition 
--------------------------------

The Proper Orthogonal Decomposition (POD) is a data-processing technique which takes a given dataset and extracts a
set of orthogonal basis functions and the corresponding coefficients.

Let us consider the solution of a numerical model of a differential equation :math:`\mathbf{u}(\mathtt{x},t)`, where
:math:`\mathtt{x} = (x,y,z)` is the position vector where the function is evaluated and :math:`t` is the time. The
idea behind the POD is to decompose the random vector field :math:`\mathbf{u}(\mathtt{x},t)`, into a set of
deterministic spatial functions :math:`\Phi_{k}{\mathtt{x}}`, multiplied by random time coefficients :math:`\alpha_{k}(t)`, so that:

.. math:: \mathbf{u}(\mathtt{x},t) =  \sum_{k=1}^{\infty}\alpha_{k}(t)\Phi_{k}(\mathtt{x})

where :math:`\Phi_{k}(\mathtt{x})` are the spatial POD modes and :math:`\alpha_{k}(t)` are the time coefficients.

The above decomposition is achieved by maximizing the energy that can be captured by the first :math:`n` spatial POD
modes (:cite:t:`POD_1`). POD modes are orthonormal and thus one can write

.. math::  \iiint_{\mathtt{x}} \Phi_{k_{1}}(\mathtt{x}) \Phi_{k_{2}}(\mathtt{x}) d\mathtt{x} = \begin{cases}
    1, & \text{if $k_1 = k_2$}.\\
    0, & \text{if $k_1 \ne k_2$}
  \end{cases}

Furthermore, at each time coefficient :math:`\alpha_{k}(t)` only depends on the spatial mode :math:`\Phi_{k}(\mathtt{x})`.
By multiplying the decomposition equation with :math:`\Phi_{k}(\mathtt{x})` and integrating over space one obtains the following

.. math:: \alpha_{k}(t) = \iiint_{\mathtt{x}} \mathbf{u}(\mathtt{x},t) \Phi_{k}(\mathtt{x}) d\mathtt{x}


The POD method has two variants, the Direct POD (:class:`.DirectPOD`) and Snapshot POD (:class:`SnapshotPOD`), both implemented
as subclasses of the parent :class:`POD` class. In cases where the dataset is large, the Snapshot POD is recommended
because it is much faster.

POD Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`POD` class is the base class for all implementations of the POD.

.. autoclass:: UQpy.dimension_reduction.pod.baseclass.POD
    :members: run


Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

    Direct POD <direct_pod>
    Snapshot POD <snapshot_pod>

Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

   POD Examples <../auto_examples/dimension_reduction/pod/index>




