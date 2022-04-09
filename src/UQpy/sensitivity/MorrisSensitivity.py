"""This module contains functionality for sensitivity analysis in ``UQpy``.

The module currently contains the following classes:

- ``Morris``: Class to compute sensitivity indices based on the Morris method.
"""
from typing import Union, Annotated

from beartype import beartype
from beartype.vale import Is

from UQpy.utilities.Utilities import process_random_state
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger, NumpyFloatArray
from UQpy.distributions import *
from UQpy.run_model.RunModel import RunModel
import numpy as np
from scipy.stats import randint


class MorrisSensitivity:
    @beartype
    def __init__(
            self,
            runmodel_object: RunModel,
            distributions: Union[JointIndependent, Union[list, tuple]],
            n_levels: Annotated[int, Is[lambda x: x >= 3]],
            delta: Union[float, int] = None,
            random_state: RandomStateType = None,
            n_trajectories: PositiveInteger = None,
            maximize_dispersion: bool = False,
    ):
        """
        Compute sensitivity indices based on the Morris screening method.

        :param runmodel_object: The computational model. It should be of type :class:`.RunModel`. The
         output QoI can be a scalar or vector of length :code:`ny`, then the sensitivity indices of all :code:`ny` outputs are
         computed independently.
        :param distributions: List of :class:`.Distribution` objects corresponding to each random variable, or
         :class:`.JointIndependent` object (multivariate RV with independent marginals).
        :param n_levels: Number of levels that define the grid over the hypercube where evaluation points are
         sampled. Must be an integer :math:`\ge 3`.
        :param delta: Size of the jump between two consecutive evaluation points, must be a multiple of delta should be
         in :code:`{1/(n_levels-1), ..., 1-1/(n_levels-1)}`.
         Default: :math:`delta=\\frac{levels\_number}{2 * (levels\_number-1)}` if `n_levels` is even,
         :math:`delta=0.5` if n_levels is odd.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
        :param n_trajectories: Number of random trajectories, usually chosen between :math:`5` and :math:`10`.
         The number of model evaluations is :code:`n_trajectories * (d+1)`. If None, the `Morris` object is created
         but not run (see :py:meth:`run` method)
        :param maximize_dispersion: If :any:`True`, generate a large number of design trajectories and keep the ones
         that maximize dispersion between all trajectories, allows for a better coverage of the input space.

         Default :any:`False`.
        """
        # Check RunModel object and distributions
        self.runmodel_object = runmodel_object
        marginals = (distributions.marginals if isinstance(distributions, JointIndependent) else distributions)
        self.icdfs = [getattr(dist, "icdf", None) for dist in marginals]
        if any(icdf is None for icdf in self.icdfs):
            raise ValueError("At least one of the distributions provided has a None icdf")
        self.dimension = len(self.icdfs)
        if self.dimension != len(self.runmodel_object.model.var_names):
            raise ValueError("The number of distributions provided does not match the number of RunModel variables")

        self.n_levels = n_levels
        self.delta = delta
        self.check_levels_delta()
        self.random_state = process_random_state(random_state)
        self.maximize_dispersion = maximize_dispersion

        self.trajectories_unit_hypercube: NumpyFloatArray = None
        """Trajectories in the unit hypercube, :class:`numpy.ndarray` of shape :code:`(n_trajectories, d+1, d)`"""
        self.trajectories_physical_space: NumpyFloatArray = None
        """Trajectories in the physical space, :class:`numpy.ndarray` of shape :code:`(n_trajectories, d+1, d)`"""
        self.elementary_effects: NumpyFloatArray = None
        """Elementary effects :math:`EE_{k}`, :class:`numpy.ndarray` of shape :code:`(n_trajectories, d, ny)`."""
        self.mustar_indices: NumpyFloatArray = None
        """First Morris sensitivity index :math:`\mu_{k}^{\star}`, :class:`numpy.ndarray` of shape :code:`(d, ny)`"""
        self.sigma_indices: NumpyFloatArray = None
        """Second Morris sensitivity index :math:`\sigma_{k}`, :class:`numpy.ndarray` of shape :code:`(d, ny)`"""

        if n_trajectories is not None:
            self.run(n_trajectories)

    def check_levels_delta(self):
        # delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}
        if (self.delta is None) and (self.n_levels % 2) == 0:
            # delta = trial_probability / (2 * (trial_probability-1))
            self.delta = self.n_levels / (2 * (self.n_levels - 1))
        elif (self.delta is None) and (self.n_levels % 2) == 1:
            self.delta = (1 / 2)  # delta = (trial_probability-1) / (2 * (trial_probability-1))
        elif not (isinstance(self.delta, (int, float)) and float(self.delta)
                  in [float(j / (self.n_levels - 1)) for j in range(1, self.n_levels - 1)]):
            raise ValueError("UQpy: delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}")

    @beartype
    def run(self, n_trajectories: PositiveInteger):
        """
        Run the Morris indices evaluation.

        The code first sample trajectories in the unit hypercube and transform them to the physical space (see method
        :py:meth:`sample_trajectories`), then runs the forward model to compute the elementary effects,
        and finally computes the sensitivity indices.

        :param n_trajectories: Number of random trajectories. Usually chosen between :math:`5` and :math:`10`.
         The number of model evaluations is :code:`n_trajectories * (d+1)`.
        """
        # Compute trajectories and elementary effects - append if any already exist
        (trajectories_unit_hypercube, trajectories_physical_space,) = \
            self.sample_trajectories(n_trajectories=n_trajectories, maximize_dispersion=self.maximize_dispersion,)
        elementary_effects = self._compute_elementary_effects(trajectories_physical_space)
        self.store_data(elementary_effects, trajectories_physical_space, trajectories_unit_hypercube)
        self.mustar_indices, self.sigma_indices = self._compute_indices(self.elementary_effects)

    def store_data(
            self,
            elementary_effects,
            trajectories_physical_space,
            trajectories_unit_hypercube,
    ):
        if self.elementary_effects is None:
            self.elementary_effects = elementary_effects
            self.trajectories_unit_hypercube = trajectories_unit_hypercube
            self.trajectories_physical_space = trajectories_physical_space
        else:
            self.elementary_effects = np.concatenate([self.elementary_effects, elementary_effects], axis=0)
            self.trajectories_unit_hypercube = np.concatenate([self.trajectories_unit_hypercube,
                                                               trajectories_unit_hypercube], axis=0)
            self.trajectories_physical_space = np.concatenate([self.trajectories_physical_space,
                                                               trajectories_physical_space], axis=0)

    @beartype
    def sample_trajectories(self, n_trajectories: PositiveInteger, maximize_dispersion: bool = False):
        """
        Create the trajectories, first in the unit hypercube then transform them in the physical space.

        :param n_trajectories: Number of random trajectories. Usually chosen between :math:`5` and :math:`10`.
         The number of model evaluations is :code:`n_trajectories * (d+1)`.
        :param maximize_dispersion: If :any:`True`, generate a large number of design trajectories and keep the ones
         that maximize dispersion between all trajectories, allows for a better coverage of the input space.
         Default :any:`False`.
        """

        trajectories_unit_hypercube = []
        perms_indices = []
        ntrajectories_all = (10 * n_trajectories if maximize_dispersion else 1 * n_trajectories)
        for r in range(ntrajectories_all):
            if self.random_state is None:
                perms = np.random.permutation(self.dimension)
            else:
                perms = self.random_state.permutation(self.dimension)
            initial_state = (1.0 / (self.n_levels - 1) *
                             randint(low=0, high=int((self.n_levels - 1) * (1 - self.delta) + 1))
                             .rvs(size=(1, self.dimension), random_state=self.random_state))
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
                des_r = np.tile(trajectories_unit_hypercube[r, :, :][np.newaxis, :, :],[self.dimension + 1, 1, 1],)
                for r2 in range(r + 1, ntrajectories_all):
                    des_r2 = np.tile(trajectories_unit_hypercube[r2, :, :][:, np.newaxis, :],
                                     [1, self.dimension + 1, 1],)
                    distances[r, r2] = np.sum(np.sqrt(np.sum((des_r - des_r2) ** 2, axis=-1)))

            # try 20000 combinations of ntrajectories trajectories, keep the one that maximizes the distance
            def compute_combi_and_dist():
                if self.random_state is None:
                    combi = np.random.choice(ntrajectories_all, replace=False, size=n_trajectories)
                else:
                    combi = self.random_state.choice(ntrajectories_all, replace=False, size=n_trajectories)
                dist_combi = 0.0
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
        r, _, d = trajectories_physical_space.shape
        # Run the model for all replicates
        elementary_effects = []
        for samples in trajectories_physical_space:
            self.runmodel_object.run(samples=samples, append_samples=False)
            qoi = np.array(self.runmodel_object.qoi_list)
            el_effect = np.zeros((self.dimension,))
            perms = [np.argwhere(bi != 0.0)[0, 0] for bi in (samples[1:] - samples[:-1])]
            for count_d, d in enumerate(perms):
                el_effect[d] = (qoi[count_d + 1] - qoi[count_d]) / self.delta
            elementary_effects.append(el_effect)
        return np.array(elementary_effects)

    @staticmethod
    def _compute_indices(elementary_effects):
        # elementary_effects is an array of shape (r, d, ny) or (r, d)
        mu_star = np.mean(np.abs(elementary_effects), axis=0)
        mu = np.mean(elementary_effects, axis=0)
        sigma = np.sqrt(np.mean((elementary_effects - mu) ** 2, axis=0))
        return mu_star, sigma
