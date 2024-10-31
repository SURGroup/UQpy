Dropout Layer Baseclass
-----------------------

This is the parent class to all Probabilistic Dropout methods as laid out by Gal et al :cite:`gal2016dropout`.
The :class:`ProbabilisticDropoutLayer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

The documentation in the :py:meth:`forward` and :py:meth:`extra_repr` on this page may be inherited from PyTorch docstrings.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.ProbabilisticDropoutLayer
    :members: forward, extra_repr
