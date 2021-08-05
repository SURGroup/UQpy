import numpy as np


class RefinedStratifiedSampling:

    def __init__(self, stratified_sampling, refinement_algorithm, samples_number=None,
                 samples_per_iteration=1, random_state=None, verbose=True):
        self.stratified_sampling = stratified_sampling
        self.samples_per_iteration = samples_per_iteration
        self.refinement_algorithm = refinement_algorithm
        self.training_points = self.stratified_sampling.samplesU01
        self.samplesU01 = self.stratified_sampling.samplesU01
        self.samples = self.stratified_sampling.samples
        self.dimension = self.samples.shape[1]
        self.random_state = random_state
        self.verbose = verbose

        self.samples_number = samples_number

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if self.samples_number is not None:
            if isinstance(self.samples_number, int) and self.samples_number > 0:
                self.run(nsamples=self.samples_number)
            else:
                raise NotImplementedError("UQpy: nsamples msut be a positive integer.")

    def run(self, nsamples):
        if isinstance(nsamples, int) and nsamples > 0:
            self.samples_number = nsamples
        else:
            raise RuntimeError("UQpy: nsamples must be a positive integer.")

        if self.samples_number <= self.samples.shape[0]:
            raise NotImplementedError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')

        initial_number = self.samples.shape[0]

        self.refinement_algorithm.initialize(self.samples_number, self.training_points)

        for i in range(initial_number, nsamples, self.samples_per_iteration):
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

