from beartype import beartype

from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *
from UQpy.utilities import Voronoi


class SimpleRefinement(Refinement):

    @beartype
    def __init__(self, strata):
        self.strata = strata

    def update_strata(self, samplesU01):
        if isinstance(self.strata, Voronoi):
            self.strata = Voronoi(seeds=samplesU01)

    def initialize(self, samples_number, training_points, samples):
        self.strata.initialize(samples_number, training_points)

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
        points_to_add = min(samples_per_iteration, samples_number - index)

        strata_metrics = self.strata.calculate_strata_metrics(index)

        bins2break = self.identify_bins(
            strata_metrics=strata_metrics,
            points_to_add=points_to_add,
            random_state=random_state,
        )

        new_points = self.strata.update_strata_and_generate_samples(dimension, points_to_add, bins2break,
                                                                    samples_u01, random_state)

        return new_points
