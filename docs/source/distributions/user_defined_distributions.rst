User-defined Distributions and Copulas
---------------------------------------------------

Defining custom distributions in :py:mod:`UQpy`. can be done by sub-classing the appropriate parent class.
The subclasses must possess the desired methods, per the parent :class:`.Distribution` class.

Custom copulas can be similarly defined by subclassing the :class:`.Copula` class and defining the appropriate methods.