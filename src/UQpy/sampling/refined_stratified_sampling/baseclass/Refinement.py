from abc import ABC, abstractmethod
import numpy as np


class Refinement(ABC):
    @abstractmethod
    def update_samples(
        self,
        samples_number,
        samples_per_iteration,
        random_state,
        index,
        dimension,
        samples_u01,
        training_points,
    ):
        pass

    def initialize(self, samples_number, training_points):
        pass

    def finalize(self, samples, samples_per_iteration):
        pass

    @staticmethod
    def identify_bins(strata_metrics, points_to_add, random_state):
        bins2break = np.array([])
        points_left = points_to_add
        while (
            np.where(strata_metrics == strata_metrics.max())[0].shape[0] < points_left
        ):
            bin = np.where(strata_metrics == strata_metrics.max())[0]
            bins2break = np.hstack([bins2break, bin])
            strata_metrics[bin] = 0
            points_left -= bin.shape[0]

        bin_for_remaining_points = random_state.choice(
            np.where(strata_metrics == strata_metrics.max())[0],
            points_left,
            replace=False,
        )
        bins2break = np.hstack([bins2break, bin_for_remaining_points])
        bins2break = list(map(int, bins2break))
        return bins2break
