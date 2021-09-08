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

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **seeds** (`ndarray`)
        Seed points for the strata. See specific subclass for definition of the seed points.

    **Methods:**
    """
    @beartype
    def __init__(self,
                 seeds: Union[None, np.ndarray] = None,
                 random_state: RandomStateType = None):

        self.seeds = seeds
        self.volume = None

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif self.random_state is None:
            self.random_state = np.random.RandomState()
        elif not isinstance(self.random_state, np.random.RandomState):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

    @abc.abstractmethod
    def stratify(self):
        """
        Perform the stratification of the unit hypercube. It is overwritten by the subclass. This method must exist in
        any subclass of the ``strata`` class.

        **Outputs/Returns:**

        The method has no returns, but it modifies the relevant attributes of the subclass.

        """

        pass

    @abc.abstractmethod
    def sample_strata(self, samples_per_stratum_number):
        pass

    @abc.abstractmethod
    def calculate_strata_metrics(self, index):
        pass

    def initialize(self, samples_number, training_points):
        pass

    def extend_weights(self, samples_per_stratum_number, index, weights):
        if int(samples_per_stratum_number[index]) != 0:
            weights.extend(
                [self.volume[index] / samples_per_stratum_number[index]] *
                int(samples_per_stratum_number[index]))
        else:
            weights.extend([0] * int(samples_per_stratum_number[index]))
