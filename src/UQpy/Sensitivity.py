# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for sensitivity analysis in ``UQpy``.

The module currently contains the following classes:

- ``Morris``: Class to compute sensitivity indices based on the Morris method.
"""

from UQpy.Distributions import *
from UQpy.RunModel import RunModel


class Morris:
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

    def __init__(self, runmodel_object, dist_object, nlevels, delta=None, random_state=None, ntrajectories=None,
                 **kwargs):

        # Check RunModel object and distributions
        self.runmodel_object = runmodel_object
        if not isinstance(self.runmodel_object, RunModel):
            raise TypeError('UQpy: runmodel_object must be an object of class RunModel')
        if isinstance(dist_object, JointInd):
            self.icdfs = [getattr(dist, 'icdf', None) for dist in dist_object.marginals]
        elif (isinstance(dist_object, (list, tuple))
              and all(isinstance(dist, Distribution) for dist in dist_object)):
            self.icdfs = [getattr(dist, 'icdf', None) for dist in dist_object]
        else:
            raise ValueError
        if any(icdf is None for icdf in self.icdfs):
            raise ValueError
        self.dimension = len(self.icdfs)
        if self.dimension != len(self.runmodel_object.var_names):
            raise ValueError

        # Check inputs nlevels and delta
        self.nlevels = nlevels
        if not isinstance(self.nlevels, int) or self.nlevels < 3:
            raise TypeError('UQpy: nlevels should be an integer >= 3')
        # delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}
        self.delta = delta
        if (self.delta is None) and (self.nlevels % 2) == 0:
            self.delta = self.nlevels / (2 * (self.nlevels - 1))  # delta = p / (2 * (p-1))
        elif (self.delta is None) and (self.nlevels % 2) == 1:
            self.delta = 1 / 2  # delta = (p-1) / (2 * (p-1))
        elif not (isinstance(self.delta, (int, float))
                  and float(self.delta) in [float(j / (self.nlevels - 1)) for j in range(1, self.nlevels - 1)]):
            raise ValueError('UQpy: delta should be in {1/(nlevels-1), ..., 1-1/(nlevels-1)}')

        # Check random state
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not (self.random_state is None or isinstance(self.random_state, np.random.RandomState)):
            raise TypeError('UQpy: random state should be None, an integer or np.random.RandomState object')

        self.kwargs = kwargs

        # Initialize the thingy
        self.trajectories_unit_hypercube = None
        self.trajectories_physical_space = None
        self.elementary_effects = None
        self.mustar_indices = None
        self.sigma_indices = None

        if ntrajectories is not None:
            self.run(ntrajectories)

    def run(self, ntrajectories):
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
        trajectories_unit_hypercube, trajectories_physical_space = self.sample_trajectories(
            ntrajectories=ntrajectories, **self.kwargs)
        elementary_effects = self._compute_elementary_effects(trajectories_physical_space)
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

        # Compute sensitivity indices
        self.mustar_indices, self.sigma_indices = self._compute_indices(self.elementary_effects)

    def sample_trajectories(self, ntrajectories, maximize_dispersion=False):
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
        from scipy.stats import randint

        trajectories_unit_hypercube = []
        perms_indices = []
        if maximize_dispersion:
            ntrajectories_all = 10 * ntrajectories
        else:
            ntrajectories_all = 1 * ntrajectories
        for r in range(ntrajectories_all):
            if self.random_state is None:
                perms = np.random.permutation(self.dimension)
            else:
                perms = self.random_state.permutation(self.dimension)
            initial_state = 1. / (self.nlevels - 1) * randint(
                low=0, high=int((self.nlevels - 1) * (1 - self.delta) + 1)).rvs(
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
                    combi = np.random.choice(ntrajectories_all, replace=False, size=ntrajectories)
                else:
                    combi = self.random_state.choice(ntrajectories_all, replace=False, size=ntrajectories)
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