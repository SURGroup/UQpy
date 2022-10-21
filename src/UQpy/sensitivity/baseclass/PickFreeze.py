import copy
from typing import Union

from beartype import beartype

from UQpy.distributions.collection import JointIndependent
from UQpy.utilities.ValidationTypes import (
    RandomStateType,
    PositiveInteger,
)


@beartype
def generate_pick_freeze_samples(
    dist_obj: Union[JointIndependent, Union[list, tuple]],
    n_samples: PositiveInteger,
    random_state: RandomStateType = None,
):

    """
    Generate samples to be used in the Pick-and-Freeze algorithm.

    **Inputs**:

    * **dist_obj** (`JointIndependent` or `list` or `tuple`):
        A distribution object or a list or tuple of distribution objects.

    * **n_samples** (`int`):
        The number of samples to be generated.

    * **random_state** (`None` or `int` or `numpy.random.RandomState`):
        A random seed or a `numpy.random.RandomState` object.

    **Outputs:**

    * **A_samples** (`ndarray`):
        Sample set A.
        Shape: `(n_samples, num_vars)`.

    * **B_samples** (`ndarray`):
        Sample set B.
        Shape: `(n_samples, num_vars)`.

    * **C_i_generator** (`generator`):
        Generator for the sample set C_i.
        Generator is used so that samples
        do not have to be stored in memory.
        C_i is a 2D array with all columns
        from B_samples, except column `i`,
        which is from A_samples.
        Shape: `(n_samples, num_vars)`.

    * **D_i_generator** (`generator`):
        Generator for the sample set C_i.
        Generator is used so that samples
        do not have to be stored in memory.
        C_i is a 2D array with all columns
        from A_samples, except column `i`,
        which is from B_samples.
        Shape: `(n_samples, num_vars)`.

    """

    # Generate samples for A and B
    samples = dist_obj.rvs(n_samples * 2, random_state=random_state)

    num_vars = samples.shape[1]

    # Split samples into two sets A and B
    A_samples = samples[:n_samples, :]
    B_samples = samples[n_samples:, :]

    # Iterator for generating C_i
    def C_i_generator():
        """Generate C_i for each i."""
        for i in range(num_vars):
            C_i = copy.deepcopy(B_samples)  #! Deepcopy so B is unchanged
            C_i[:, i] = A_samples[:, i]
            yield C_i

    # Iterator for generating D_i
    def D_i_generator():
        """Generate D_i for each i."""
        for i in range(num_vars):
            D_i = copy.deepcopy(A_samples)  #! Deepcopy so A is unchanged
            D_i[:, i] = B_samples[:, i]
            yield D_i

    return A_samples, B_samples, C_i_generator(), D_i_generator()
