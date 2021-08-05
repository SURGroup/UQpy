from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *


class SimpleRefinement(Refinement):

    def __init__(self, strata):
        self.strata = strata
        self.verbose = True

    def initialize(self, samples_number, training_points):
        self.strata.initialize(samples_number, training_points)

    def update_samples(self, samples_number, samples_per_iteration,
                       random_state, index, dimension, samplesU01, training_points):
        points_to_add = min(samples_per_iteration, samples_number - index)

        strata_metrics = self.strata.calculate_strata_metrics(index)

        bins2break = self.identify_bins(strata_metrics=strata_metrics,
                                        points_to_add=points_to_add,
                                        random_state=random_state)

        new_points = self.strata\
            .update_strata_and_generate_samples(dimension, points_to_add,
                                                bins2break, samplesU01, random_state)

        return new_points
