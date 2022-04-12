from beartype import beartype

from UQpy.sampling.stratified_sampling.refinement.baseclass.Refinement import *
from UQpy.sampling.stratified_sampling.strata.VoronoiStrata import VoronoiStrata


class RandomRefinement(Refinement):

    @beartype
    def __init__(self, strata):
        """
        Randomized refinement algorithm. Strata to be refined are selected at random according to their probability
        weights.

        :param strata: :class:`.Strata` object containing already stratified domain to be adaptively sampled using
         :class:`.RefinedStratifiedSampling`
        """
        self.strata = strata

    def update_strata(self, samplesU01):
        if isinstance(self.strata, VoronoiStrata):
            self.strata = VoronoiStrata(seeds=samplesU01)

    def initialize(self, samples_number, training_points, samples):
        self.strata.initialize(samples_number, training_points)

    def update_samples(
        self,
        nsamples,
        samples_per_iteration,
        random_state,
        index,
        dimension,
        samples_u01,
        training_points,
    ):
        points_to_add = min(samples_per_iteration, nsamples - index)

        strata_metrics = self.strata.calculate_strata_metrics(index)

        bins2break = self.identify_bins(
            strata_metrics=strata_metrics,
            points_to_add=points_to_add,
            random_state=random_state,
        )

        new_points = self.strata.update_strata_and_generate_samples(dimension, points_to_add, bins2break,
                                                                    samples_u01, random_state)

        return new_points
