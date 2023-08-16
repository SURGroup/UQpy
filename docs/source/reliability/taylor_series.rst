Taylor Series
-------------

:class:`.TaylorSeries` is a class that calculates the reliability  of a model using the First Order Reliability Method
(FORM), Inverse First Order Reliability Method (InverseFORM) or Second Order Reliability Method (SORM) based on the
first-order and second-order Taylor series expansion approximation of the performance function, respectively
(:cite:`TaylorSeries1`, :cite:`FORM_XDu`, :cite:`TaylorSeries2`).

.. image:: ../_static/Reliability_FORM.png
   :scale: 40 %
   :alt:  Graphical representation of the FORM.
   :align: center

The :class:`.TaylorSeries` class is the parent class of the :class:`.FORM`, :class:`InverseFORM`, and :class:`.SORM`
classes that perform the FORM, InverseFORM, and SORM, respectively.
These classes can be imported in a python script using the following command:

>>> from UQpy.reliability.taylor_series import FORM, InverseFORM, SORM


.. toctree::
   :maxdepth: 1

    FORM <form>
    InverseFORM <inverse_form>
    SORM <sorm>





