import logging
from typing import Union

from UQpy.sampling.latin_hypercube_criteria import Random
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.distributions import *
from UQpy.sampling.latin_hypercube_criteria.baseclass.Criterion import *
import numpy as np
from UQpy.distributions import DistributionContinuous1D, JointIndependent


class LatinHypercubeSampling:

    """
    Perform Latin hypercube sampling (LHS) of random variables.

    **Input:**

    * **distributions** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

        All distributions in ``LatinHypercubeSampling`` must be independent. ``LatinHypercubeSampling`` does not
        generate correlated random variables. Therefore, for multi-variate designs the `dist_object` must be a list of
        ``DistributionContinuous1D`` objects or an object of the ``JointInd`` class.

    * **samples_number** (`int`):
        Number of samples to be drawn from each distribution.

    * **criterion** (`Criterion`):
        The criterion for pairing the generating sample points
            Options:
                1. 'Random' - completely random. \n
                2. 'Centered' - points only at the centre. \n
                3. 'MaxiMin' - maximizing the minimum distance between points. \n
                4. 'MinCorrelation' - minimizing the correlation between the points. \n
                5. User-defined criterion class.

    **Attributes:**

    * **samples** (`ndarray`):
        The generated LHS samples.

    * **samplesU01** (`ndarray`):
        The generated LHS samples on the unit hypercube.

    **Methods**

    """

    @beartype
    def __init__(
        self,
        distributions: Union[Distribution, list[Distribution]],
        samples_number: PositiveInteger,
        criterion: Criterion = Random(),
    ):

        self.dist_object = distributions
        self.criterion = criterion
        self.samples_number = samples_number
        self.logger = logging.getLogger(__name__)

        if isinstance(self.dist_object, list):
            self.samples = np.zeros([self.samples_number, len(self.dist_object)])
        elif isinstance(self.dist_object, DistributionContinuous1D):
            self.samples = np.zeros([self.samples_number, 1])
        elif isinstance(self.dist_object, JointIndependent):
            self.samples = np.zeros(
                [self.samples_number, len(self.dist_object.marginals)]
            )

        self.samplesU01 = np.zeros_like(self.samples)

        if self.samples_number is not None:
            self.run(self.samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):

        """
        Execute the random sampling in the ``LatinHypercubeSampling`` class.

        The ``run`` method is the function that performs random sampling in the ``LatinHypercubeSampling`` class. If
        `samples_number` is provided, the ``run`` method is automatically called when the ``LatinHypercubeSampling``
        object is defined. The user may also call the ``run`` method directly to generate samples. The ``run`` method of
        the ``LatinHypercubeSampling`` class cannot be invoked multiple times for sample size extension.

        **Input:**

        * **samples_number** (`int`):
            Number of samples to be drawn from each distribution.

            If the ``run`` method is invoked multiple times, the newly generated samples will overwrite the existing
            samples.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` and `samplesU01` attributes
        of the ``LatinHypercubeSampling`` object.

        """

        self.samples_number = samples_number
        self.logger.info("UQpy: Running Latin Hypercube sampling...")
        self.criterion.create_bins(self.samples)

        u_lhs = self.criterion.generate_samples()
        self.samplesU01 = u_lhs

        if isinstance(self.dist_object, list):
            for j in range(len(self.dist_object)):
                if hasattr(self.dist_object[j], "icdf"):
                    self.samples[:, j] = self.dist_object[j].icdf(u_lhs[:, j])

        elif isinstance(self.dist_object, JointIndependent):
            if all(hasattr(m, "icdf") for m in self.dist_object.marginals):
                for j in range(len(self.dist_object.marginals)):
                    self.samples[:, j] = self.dist_object.marginals[j].icdf(u_lhs[:, j])

        elif isinstance(self.dist_object, DistributionContinuous1D):
            if hasattr(self.dist_object, "icdf"):
                self.samples = self.dist_object.icdf(u_lhs)

        self.logger.info("Successful execution of LHS design.")
