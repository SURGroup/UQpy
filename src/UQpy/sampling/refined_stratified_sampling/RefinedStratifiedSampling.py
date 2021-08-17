import logging

import numpy as np
from beartype import beartype
from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *
from UQpy.sampling.StratifiedSampling import *
from UQpy.utilities.ValidationTypes import RandomStateType,PositiveInteger
from UQpy.utilities.Utilities import process_random_state



class RefinedStratifiedSampling:

    @beartype
    def __init__(self,
                 stratified_sampling: StratifiedSampling,
                 refinement_algorithm: Refinement,
                 samples_number: PositiveInteger = None,
                 samples_per_iteration: int = 1,
                 random_state: RandomStateType = None):
        self.stratified_sampling = stratified_sampling
        self.samples_per_iteration = samples_per_iteration
        self.refinement_algorithm = refinement_algorithm
        self.training_points = self.stratified_sampling.samplesU01
        self.samplesU01 = self.stratified_sampling.samplesU01
        self.samples = self.stratified_sampling.samples
        self.dimension = self.samples.shape[1]
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        self.samples_number = samples_number

        self.random_state = process_random_state(random_state)

        if self.samples_number is not None:
            self.run(nsamples=self.samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):
        self.samples_number = samples_number

        if self.samples_number <= self.samples.shape[0]:
            raise ValueError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')

        initial_number = self.samples.shape[0]

        self.refinement_algorithm.initialize(self.samples_number, self.training_points)

        for i in range(initial_number, samples_number, self.samples_per_iteration):
            new_points = self.refinement_algorithm\
                .update_samples(self.samples_number, self.samples_per_iteration,
                                self.random_state, i, self.dimension, self.samplesU01,
                                self.training_points)
            self.append_samples(new_points)

            self.refinement_algorithm.finalize(self.samples, self.samples_per_iteration)

    def append_samples(self, new_points):
        # Adding new sample to training points, samplesU01 and samples attributes
        self.training_points = np.vstack([self.training_points, new_points])
        self.samplesU01 = np.vstack([self.samplesU01, new_points])
        new_point_ = np.zeros_like(new_points)
        for k in range(self.dimension):
            new_point_[:, k] = self.stratified_sampling.distributions[k].icdf(new_points[:, k])
        self.samples = np.vstack([self.samples, new_point_])

