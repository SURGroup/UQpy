from typing import Union
import numpy as np
import abc
from beartype import beartype
from UQpy.utilities.ValidationTypes import RandomStateType


class Strata:
    """
    Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling strata.

    This is the parent class for all spatial stratified_sampling. This parent class only provides the framework for
    stratification and cannot be used directly for the stratification. Stratification is done by calling the child
    class for the desired stratification.


    **Inputs:**

    * **seeds** (`ndarray`)
        Define the seed points for the strata. See specific subclass for definition of the seed points.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **seeds** (`ndarray`)
        Seed points for the strata. See specific subclass for definition of the seed points.

    **Methods:**
    """

    @beartype
    def __init__(self, seeds: Union[None, np.ndarray] = None):

        self.seeds = seeds
        self.volume = None

    @abc.abstractmethod
    def stratify(self, random_state):
        """
        Perform the stratification of the unit hypercube. It is overwritten by the subclass. This method must exist in
        any subclass of the ``strata`` class.

        **Outputs/Returns:**

        The method has no returns, but it modifies the relevant attributes of the subclass.

        """

        pass

    @abc.abstractmethod
    def sample_strata(self, samples_per_stratum_number, random_state):
        pass

    @abc.abstractmethod
    def calculate_strata_metrics(self, index):
        pass

    def initialize(self, samples_number, training_points):
        pass

    def extend_weights(self, samples_per_stratum_number, index, weights):
        if int(samples_per_stratum_number[index]) != 0:
            weights.extend(
                [self.volume[index] / samples_per_stratum_number[index]]
                * int(samples_per_stratum_number[index])
            )
        else:
            weights.extend([0] * int(samples_per_stratum_number[index]))
