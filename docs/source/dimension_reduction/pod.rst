Proper Orthogonal Decomposition 
--------------------------------

The Proper Orthogonal Decomposition (POD) is a post-processing technique which takes a given dataset and extracts a
set of orthogonal basis functions and the corresponding coefficients. The idea of this method, is to analyze large
amounts of data in order to gain a better understanding of the simulated processes and reduce noise. POD method has
two variants, the Direct POD and Snapshot POD. In cases where the dataset is large, the Snapshot POD is recommended as
it is much faster.

POD Baseclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: UQpy.dimension_reduction.pod.baseclass.POD
    :members: run

Examples
""""""""""

.. toctree::

   POD Examples <../auto_examples/dimension_reduction/pod/index>

Implementations
""""""""""""""""
.. toctree::
   :maxdepth: 1

    Direct POD <direct_pod>
    Snapshot POD <snapshot_pod>


