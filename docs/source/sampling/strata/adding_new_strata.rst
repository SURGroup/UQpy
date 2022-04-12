Adding a new Strata class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding a new type of stratification requires creating a new subclass of the :class:`.Strata` class that defines the
desired geometric decomposition. This subclass must have a :meth:`stratify` method that overwrites the corresponding
method in the parent class and performs the stratification.