"""This module contains functionality for sensitivity analysis in ``UQpy``.

The module currently contains the following classes:

- ``Morris``: Class to compute sensitivity indices based on the Morris method.
"""
from typing import Union, Annotated

from beartype import beartype
from beartype.vale import Is

from UQpy.utilities.Utilities import process_random_state
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger
from UQpy.distributions import *
from UQpy.RunModel import RunModel
import numpy as np
from scipy.stats import randint


class MorrisSensitivity:
    """
    Compute sensitivity indices based on the Morris screening method.

    **Inputs:**

    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type ``RunModel`` (see ``RunModel`` class). The output QoI can be a
        scalar or vector of length `ny`, then the sensitivity indices of all `ny` outputs are computed independently.

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable, or ``JointInd`` object (multivariate RV
        with independent marginals).

    * **nlevels** (`int`):
        Number of levels that define the grid over the hypercube where evaluation points are sampled. Must be an
        integer >= 3.

    * **delta** (`float`):
        Size of the jump between two consecutive evaluation points, must be a multiple of delta should be in
        `{1/(nlevels-1), ..., 1-1/(nlevels-1)}`.

        Default: :math:`delta=\\frac{nlevels}{2 * (nlevels-1)}` if nlevels is even, delta=0.5 if nlevels is odd.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

    * **ntrajectories** (`int`):
        Number of random trajectories, usually chosen between 5 and 10. The number of model evaluations is
        `ntrajectories * (d+1)`. If None, the `Morris` object is created but not run (see `run` method)

    * **kwargs**:
        Additional key-word arguments transferred to the ``sample_trajectories`` method that samples trajectories where
        to evaluate the model.

    **Attributes:**

    * **elementary_effects** (`ndarray`):
        Elementary effects :math:`EE_{k}`, `ndarray` of shape `(ntrajectories, d, ny)`.

    * **mustar_indices** (`ndarray`):
        First Morris sensitivity index :math:`\mu_{k}^{\star}`, `ndarray` of shape `(d, ny)`

    * **sigma_indices** (`ndarray`):
        Second Morris sensitivity index :math:`\sigma_{k}`, `ndarray` of shape `(d, ny)`

    * **trajectories_unit_hypercube** (`ndarray`):
        Trajectories in the unit hypercube, `ndarray` of shape `(ntrajectories, d+1, d)`

    * **trajectories_physical_space** (`ndarray`):
        Trajectories in the physical space, `ndarray` of shape `(ntrajectories, d+1, d)`

    **Methods:**
    """
    @beartype
    def __init__(self,
                 runmodel_object: RunModel,
                 distributions: Union[JointIndependent, Union[list, tuple]],
                 levels_number: Annotated[int, Is[lambda x: x >= 3]],
                 delta: Union[float, int] = None,
                 random_state: RandomStateType = None,
                 trajectories_number: PositiveInteger = None,
                 maximize_dispersion: bool = False):

        # Check RunModel object and distributions
        self.runmodel_object = runmodel_object
        marginals = distributions.marginals if isinstance(distributions, JointIndependent) else distributions
        self.icdfs = [getattr(dist, 'icdf', None) for dist in marginals]
        if any(icdf is None for icdf in self.icdfs):
            raise ValueError("At least one of the distributions provided has a None icdf")
        self.dimension = len(self.icdfs)
        if self.dimension != len(self.runmodel_object.var_names):
            raise ValueError("The number of distributions provided does not match the number of RunModel variables")

        self.levels_number = levels_number
        self.delta = delta
        self.check_levels_delta()
        self.random_state = process_random_state(random_state)
        self.maximize_dispersion = maximize_dispersion

        self.trajectories_unit_hypercube = None
        self.trajectories_physical_space = None
        self.elementary_effects = None
        self.mustar_indices = None
        self.sigma_indices = None

        if trajectories_number is not None:
            self.run(trajectories_number)

    def check_levels_delta(self):
        # delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}
        if (self.delta is None) and (self.levels_number % 2) == 0:
            # delta = trial_probability / (2 * (trial_probability-1))
            self.delta = self.levels_number / (2 * (self.levels_number - 1))
        elif (self.delta is None) and (self.levels_number % 2) == 1:
            self.delta = 1 / 2  # delta = (trial_probability-1) / (2 * (trial_probability-1))
        elif not (isinstance(self.delta, (int, float))
                  and float(self.delta) in
                  [float(j / (self.levels_number - 1)) for j in range(1, self.levels_number - 1)]):
            raise ValueError('UQpy: delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}')

    @beartype
    def run(self, trajectories_number: PositiveInteger):
        """
        Run the Morris indices evaluation.

        The code first sample trajectories in the unit hypercube and transform them to the physical space (see method
        `sample_trajectories`), then runs the forward model to compute the elementary effects (method
        `_compute_elementary_effects`), and finally computes the sensitivity indices (method `_compute_indices`).

        **Inputs:**

        * **ntrajectories** (`int`):
            Number of random trajectories. Usually chosen between 5 and 10. The number of model evaluations is
            `ntrajectories * (d+1)`.


        """
        # Compute trajectories and elementary effects - append if any already exist
        trajectories_unit_hypercube, trajectories_physical_space = \
            self.sample_trajectories(trajectories_number=trajectories_number,
                                     maximize_dispersion=self.maximize_dispersion)
        elementary_effects = self._compute_elementary_effects(trajectories_physical_space)
        self.store_data(elementary_effects, trajectories_physical_space, trajectories_unit_hypercube)
        self.mustar_indices, self.sigma_indices = self._compute_indices(self.elementary_effects)

    def store_data(self, elementary_effects, trajectories_physical_space, trajectories_unit_hypercube):
        if self.elementary_effects is None:
            self.elementary_effects = elementary_effects
            self.trajectories_unit_hypercube = trajectories_unit_hypercube
            self.trajectories_physical_space = trajectories_physical_space
        else:
            self.elementary_effects = np.concatenate([self.elementary_effects, elementary_effects], axis=0)
            self.trajectories_unit_hypercube = np.concatenate(
                [self.trajectories_unit_hypercube, trajectories_unit_hypercube], axis=0)
            self.trajectories_physical_space = np.concatenate(
                [self.trajectories_physical_space, trajectories_physical_space], axis=0)

    @beartype
    def sample_trajectories(self, trajectories_number: PositiveInteger, maximize_dispersion: bool = False):
        """
        Create the trajectories, first in the unit hypercube then transform them in the physical space.

        * **ntrajectories** (`int`):
            Number of random trajectories. Usually chosen between 5 and 10. The number of model evaluations is
            `ntrajectories * (d+1)`.

        * **maximize_dispersion** (`bool`):
            If True, generate a large number of design trajectories and keep the ones that maximize dispersion between
            all trajectories, allows for a better coverage of the input space.

            Default False.
        """
        trajectories_unit_hypercube = []
        perms_indices = []
        ntrajectories_all = 10 * trajectories_number if maximize_dispersion else 1 * trajectories_number
        for r in range(ntrajectories_all):
            if self.random_state is None:
                perms = np.random.permutation(self.dimension)
            else:
                perms = self.random_state.permutation(self.dimension)
            initial_state = 1. / (self.levels_number - 1) * randint(
                low=0, high=int((self.levels_number - 1) * (1 - self.delta) + 1)).rvs(
                size=(1, self.dimension), random_state=self.random_state)
            trajectory_uh = np.tile(initial_state, [self.dimension + 1, 1])
            for count_d, d in enumerate(perms):
                trajectory_uh[count_d + 1:, d] = initial_state[0, d] + self.delta
            trajectories_unit_hypercube.append(trajectory_uh)
            perms_indices.append(perms)
        trajectories_unit_hypercube = np.array(trajectories_unit_hypercube)  # ndarray (r, d+1, d)

        # if maximize_dispersion, compute the 'best' trajectories
        if maximize_dispersion:
            from itertools import combinations
            distances = np.zeros((ntrajectories_all, ntrajectories_all))
            for r in range(ntrajectories_all):
                des_r = np.tile(trajectories_unit_hypercube[r, :, :][np.newaxis, :, :], [self.dimension + 1, 1, 1])
                for r2 in range(r + 1, ntrajectories_all):
                    des_r2 = np.tile(trajectories_unit_hypercube[r2, :, :][:, np.newaxis, :],
                                     [1, self.dimension + 1, 1])
                    distances[r, r2] = np.sum(np.sqrt(np.sum((des_r - des_r2) ** 2, axis=-1)))

            # try 20000 combinations of ntrajectories trajectories, keep the one that maximizes the distance
            def compute_combi_and_dist():
                if self.random_state is None:
                    combi = np.random.choice(ntrajectories_all, replace=False, size=trajectories_number)
                else:
                    combi = self.random_state.choice(ntrajectories_all, replace=False, size=trajectories_number)
                dist_combi = 0.
                for pairs in list(combinations(combi, 2)):
                    dist_combi += distances[min(pairs), max(pairs)] ** 2
                return combi, np.sqrt(dist_combi)

            comb_to_keep, dist_comb = compute_combi_and_dist()
            for _ in range(1, 20000):
                comb, new_dist_comb = compute_combi_and_dist()
                if new_dist_comb > dist_comb:
                    comb_to_keep, dist_comb = comb, new_dist_comb
            trajectories_unit_hypercube = np.array([trajectories_unit_hypercube[j] for j in comb_to_keep])

        # Avoid 0 and 1 cdf values
        trajectories_unit_hypercube[trajectories_unit_hypercube < 0.01] = 0.01
        trajectories_unit_hypercube[trajectories_unit_hypercube > 0.99] = 0.99

        # Transform to physical space via icdf
        trajectories_physical_space = []
        for trajectory_uh in trajectories_unit_hypercube:
            trajectory_ps = np.zeros_like(trajectory_uh)
            for count_d, (design_d, icdf_d) in enumerate(zip(trajectory_uh.T, self.icdfs)):
                trajectory_ps[:, count_d] = icdf_d(x=design_d)
            trajectories_physical_space.append(trajectory_ps)
        trajectories_physical_space = np.array(trajectories_physical_space)
        return trajectories_unit_hypercube, trajectories_physical_space

    def _compute_elementary_effects(self, trajectories_physical_space):
        """
        Compute elementary effects from the trajectories in the physical space.
        """
        r, _, d = trajectories_physical_space.shape
        # Run the model for all replicates
        elementary_effects = []
        for samples in trajectories_physical_space:
            self.runmodel_object.run(samples=samples, append_samples=False)
            qoi = np.array(self.runmodel_object.qoi_list)
            el_effect = np.zeros((self.dimension,))
            perms = [np.argwhere(bi != 0.)[0, 0] for bi in (samples[1:] - samples[:-1])]
            for count_d, d in enumerate(perms):
                el_effect[d] = (qoi[count_d + 1] - qoi[count_d]) / self.delta
            elementary_effects.append(el_effect)
        return np.array(elementary_effects)

    @staticmethod
    def _compute_indices(elementary_effects):
        """
        Compute indices from elementary effects.
        """
        # elementary_effects is an array of shape (r, d, ny) or (r, d)
        mu_star = np.mean(np.abs(elementary_effects), axis=0)
        mu = np.mean(elementary_effects, axis=0)
        sigma = np.sqrt(np.mean((elementary_effects - mu) ** 2, axis=0))
        return mu_star, sigma
