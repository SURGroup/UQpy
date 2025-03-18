Fourier Neural Operator (FNO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of the Fourier neural operator (FNO) as defined by Li et al. :cite:`li2021fno`
There is no explict Fourier neural operator class, but rather an implementation of the Fourier layers defined by Li in
Figure 2. This module provides Fourier layers for 1d, 2d, and 3d signals that compute the Fast Fourier Transform
in their respective number of dimensions.

Schematically, we represent the Fourier Neural Operator as having three parts shown below.
After the input layer, we lift the number of features up to the width of the Fourier layers, apply the Fourier layers,
then project the Fourier representation to the desired number of out channels. See the example below for details.

.. figure:: ./figures/fourier_network_diagram.pdf
   :align: center
   :class: with-border
   :width: 800
   :alt: A diagram showing the architecture of a Fourier neural operator.

   The architecture of a generic Fourier neural operator.


Examples
--------


.. toctree::

   Burgers' Equation <../../auto_examples/scientific_machine_learning/fourier_neural_operator/burgers>