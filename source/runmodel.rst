.. _runmodel:

Simulations
============

In ``UQpy``, forward simulations are initiated via the ``RunModel`` module. This module can interact with Python computational models as well as third-party software, allowing great flexibility in the definition of the forward model.

An object of the class ``RunModel`` is instantiated as follows::

	>>> from UQpy.RunModel import RunModel
	