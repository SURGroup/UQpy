import logging

import numpy as np

from UQpy.utilities.strata.baseclass.Strata import Strata
from UQpy.utilities.strata.StratificationCriterion import StratificationCriterion
import scipy.stats as stats


class Rectangular(Strata):
    def __init__(self, strata_number=None, input_file=None, seeds=None, widths=None, random_state=None,
                 stratification_criterion=StratificationCriterion.RANDOM):
        super().__init__(seeds=seeds, random_state=random_state)

        self.logger = logging.getLogger(__name__)
        self.input_file = input_file
        self.strata_number = strata_number
        self.widths = widths
        self.stratification_criterion = stratification_criterion
        self.stratify()

    def stratify(self):
        """
        Performs the rectangular stratification.
        """
        self.logger.info('UQpy: Creating Rectangular stratification ...')

        if self.strata_number is None:
            if self.input_file is None:
                if self.widths is None or self.seeds is None:
                    raise RuntimeError('UQpy: The strata are not fully defined. Must provide `n_strata`, `input_file`, '
                                       'or `seeds` and `widths`.')

            else:
                # Read the strata from the specified input file
                # See documentation for input file formatting
                array_tmp = np.loadtxt(self.input_file)
                self.seeds = array_tmp[:, 0:array_tmp.shape[1] // 2]
                self.widths = array_tmp[:, array_tmp.shape[1] // 2:]

                # Check to see that the strata are space-filling
                space_fill = np.sum(np.prod(self.widths, 1))
                if 1 - space_fill > 1e-5:
                    raise RuntimeError('UQpy: The stratum design is not space-filling.')
                if 1 - space_fill < -1e-5:
                    raise RuntimeError('UQpy: The stratum design is over-filling.')

        # Define a rectilinear stratification by specifying the number of strata in each dimension via nstrata
        else:
            self.seeds = np.divide(self.fullfact(self.strata_number), self.strata_number)
            self.widths = np.divide(np.ones(self.seeds.shape), self.strata_number)

        self.volume = np.prod(self.widths, axis=1)

        self.logger.info('UQpy: Rectangular stratification created.')

    @staticmethod
    def fullfact(levels):

        """
        Create a full-factorial design

        Note: This function has been modified from pyDOE, released under BSD License (3-Clause)\n
        Copyright (C) 2012 - 2013 - Michael Baudin\n
        Copyright (C) 2012 - Maria Christopoulou\n
        Copyright (C) 2010 - 2011 - INRIA - Michael Baudin\n
        Copyright (C) 2009 - Yann Collette\n
        Copyright (C) 2009 - CEA - Jean-Marc Martinez\n
        Original source code can be found at:\n
        https://pythonhosted.org/pyDOE/#\n
        or\n
        https://pypi.org/project/pyDOE/\n
        or\n
        https://github.com/tisimst/pyDOE/\n

        **Input:**

        * **levels** (`list`):
            A list of integers that indicate the number of levels of each input design factor.

        **Output:**

        * **ff** (`ndarray`):
            Full-factorial design matrix.
        """
        # Number of factors
        n_factors = len(levels)
        # Number of combinations
        n_comb = np.prod(levels)
        ff = np.zeros((n_comb, n_factors))

        level_repeat = 1
        range_repeat = np.prod(levels)
        for i in range(n_factors):
            range_repeat //= levels[i]
            lvl = []
            for j in range(levels[i]):
                lvl += [j] * level_repeat
            rng = lvl * range_repeat
            level_repeat *= levels[i]
            ff[:, i] = rng

        return ff

    def plot_2d(self):
        """
        Plot the rectangular stratification.

        This is an instance method of the ``RectangularStrata`` class that can be called to plot the boundaries of a
        two-dimensional ``RectangularStrata`` object on :math:`[0, 1]^2`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig = plt.figure()
        ax = fig.gca()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        for i in range(self.seeds.shape[0]):
            rect1 = patches.Rectangle(self.seeds[i, :], self.widths[i, 0], self.widths[i, 1], linewidth=1,
                                      edgecolor='b', facecolor='none')
            ax.add_patch(rect1)

        return fig

    def sample_strata(self, samples_per_stratum_number):
        samples_in_strata, weights = [], []
        for i in range(self.seeds.shape[0]):
            samples_temp = np.zeros([int(samples_per_stratum_number[i]), self.seeds.shape[1]])
            for j in range(self.seeds.shape[1]):
                if self.stratification_criterion == StratificationCriterion.RANDOM:
                    samples_temp[:, j] = stats.uniform.rvs(loc=self.seeds[i, j],
                                                           scale=self.widths[i, j],
                                                           random_state=self.random_state,
                                                           size=int(samples_per_stratum_number[i]))
                else:
                    samples_temp[:, j] = self.seeds[i, j] + self.widths[i, j] / 2.

            samples_in_strata.append(samples_temp)
            self.extend_weights(samples_per_stratum_number, i, weights)
        return samples_in_strata, weights

    def calculate_strata_metrics(self, index):
        s = np.zeros(index)
        for i in range(index):
            s[i] = self.volume[i] ** 2
        return s

    def calculate_gradient_strata_metrics(self, index, dy_dx):
        dy_dx1 = dy_dx[:index]
        stratum_variance = (1 / 12) * self.widths ** 2
        s = np.zeros(index)
        for i in range(index):
            s[i] = np.sum(dy_dx1[i, :] * stratum_variance[i, :] * dy_dx1[i, :]) * self.volume[i] ** 2
        return s

    def estimate_gradient(self, index, samples_u01, training_points, qoi, max_train_size=None):
        return super().estimate_gradient(np.atleast_2d(training_points),
                                         np.atleast_2d(np.array(qoi)),
                                         self.seeds + 0.5 * self.widths)

    def update_strata_and_generate_samples(self, dimension, points_to_add, bins2break, samples_u01, random_state):
        new_points = np.zeros([points_to_add, dimension])
        for j in range(len(bins2break)):
            new_points[j, :] = self._update_stratum_and_generate_sample(bins2break[j], samples_u01, random_state)
        return new_points

    def _update_stratum_and_generate_sample(self, bin_, samples_u01, random_state):
        # Cut the stratum in the direction of maximum length
        cut_dir_temp = self.widths[bin_, :]
        dir2break = np.random.choice(np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))[0])

        # Divide the stratum bin2break in the direction dir2break
        self.widths[bin_, dir2break] = self.widths[bin_, dir2break] / 2
        self.widths = np.vstack([self.widths, self.widths[bin_, :]])
        self.seeds = np.vstack([self.seeds, self.seeds[bin_, :]])
        if samples_u01[bin_, dir2break] < self.seeds[bin_, dir2break] + \
                self.widths[bin_, dir2break]:
            self.seeds[-1, dir2break] = self.seeds[bin_, dir2break] + \
                                                      self.widths[bin_, dir2break]
        else:
            self.seeds[bin_, dir2break] = self.seeds[bin_, dir2break] + \
                                                        self.widths[bin_, dir2break]

        self.volume[bin_] = self.volume[bin_] / 2
        self.volume = np.append(self.volume, self.volume[bin_])

        # Add a uniform random sample inside the new stratum
        new_samples = stats.uniform.rvs(loc=self.seeds[-1, :], scale=self.widths[-1, :],
                                        random_state=random_state)

        return new_samples
