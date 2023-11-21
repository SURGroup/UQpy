Physics-informed Polynomial Chaos Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Polynomial chaos expansion can be used in physics-informed machine learning as an efficient surrogate model allowing for
analytical uncertainty quantification. The PCE constrained to adhere to the known physics of the model (referenced as
(PC :math:`^2`), combines the conventional experimental design with additional constraints from the physics of the model.
The constraints are represented by set of differential equations and specified boundary conditions.
PC :math:`^2` framework implemented in UQPy consists of three classes.

PdeData class
"""""""""""""""""""""""""""""""""""

The first class :class:`.PdeData` contains general physical information (geometry, boundary conditions) describing the
governing differential equation. It is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData


.. autoclass:: UQpy.surrogates.polynomial_chaos.physics_informed.PdeData.PdeData
    :members:

PdePce class
"""""""""""""""""""""""""""""""""""

The second class in the PC :math:`^2` framework is :class:`.PdePCE` containing PDE physical data and definitions of PDE
in PCE context. The class is imported using the following command:


>>> from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import PdePCE


.. autoclass:: UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE.PdePCE
    :members:

ConstrainedPCE class
"""""""""""""""""""""""""""""""""""

Finally, a numerical solvers based on Karush-Kuhn-Tucker normal equations are defined in  the :class:`.ConstrainedPCE`
imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE import ConstrainedPCE


.. autoclass:: UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE.ConstrainedPCE
    :members:




ReducedPCE Class
"""""""""""""""""""""""""""""""""""
Once the PC :math:`^2` is created, it can be easily exploited for UQ as standard PCE. However, since differential
equations are typically defined in physical space and thus the PCE contains also deterministic space-time variables.
Their influence can be filtered out by the :class:`.ReducedPCE` class.


The :class:`.ReducedPCE` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.physics_informed.ReducedPCE import ReducedPCE

.. autoclass:: UQpy.surrogates.polynomial_chaos.physics_informed.ReducedPCE.ReducedPCE
    :members:

Examples
""""""""""

.. toctree::

   Polynomial Chaos Expansion Examples <../../auto_examples/surrogates/pce/index>