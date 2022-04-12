Distributions ND
----------------------------------

In :py:mod:`UQpy`, multivariate distributions inherit from the :class:`.DistributionND` class


:py:mod:`UQpy` has some inbuilt multivariate distributions, which are directly child classes of :class:`.DistributionND`.
Additionally, joint distributions can be built from their marginals through the use of the :class:`.JointIndependent` and
:class:`.JointCopula` classes described below.


.. toctree::
   :maxdepth: 1

    List of multivariate distributions <multivariate_distributions>
    Joint from independent marginals <joint_from_independent>
    Joint from marginals and copula <joint_from_marginals_copula>
    Examples <../auto_examples/distributions/multivariate/index>



