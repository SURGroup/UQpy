from UQpy.sampling.refined_stratified_sampling.baseclass.Refinement import *


class GradientEnhancedRefinement(Refinement):

    def __init__(self, strata, runmodel_object, surrogate,
                 nearest_points_number=None,
                 qoi_name=None, step_size=0.005):
        self.runmodel_object = runmodel_object
        self.step_size = step_size
        self.nearest_points_number = nearest_points_number
        self.qoi_name = qoi_name
        self.strata = strata
        self.dy_dx = 0

        if self.runmodel_object is not None:
            if type(self.runmodel_object).__name__ not in ['RunModel']:
                raise NotImplementedError("UQpy Error: runmodel_object must be an object of the RunModel class.")

        if surrogate is not None:
            if surrogate is not None and hasattr(surrogate, 'fit') and hasattr(surrogate, 'predict'):
                self.surrogate = surrogate
            else:
                raise NotImplementedError("UQpy Error: krig_object must have 'fit' and 'predict' methods.")

    def initialize(self, nsamples, training_points):
        self.dy_dx = np.zeros((nsamples, np.size(training_points[1])))
        self.strata.initialize(nsamples, training_points)

    def update_samples(self, samples_number, samples_per_iteration,
                       random_state, index, dimension, samples_u01, training_points):
        points_to_add = min(samples_per_iteration, samples_number - index)

        gradient_values = self.strata.estimate_gradient(index, samples_u01, training_points, self._convert_qoi_tolist())

        strata_metrics = self.strata\
            .calculate_gradient_strata_metrics(index, gradient_values)

        bins2break = self.identify_bins(strata_metrics=strata_metrics,
                                        points_to_add=points_to_add,
                                        random_state=random_state)

        new_points = self.strata.update_strata_and_generate_samples(dimension, points_to_add, bins2break,
                                                                    samples_u01, random_state)

        return new_points

    def finalize(self, samples, samples_per_iteration):
        self.runmodel_object.run(samples=np.atleast_2d(samples[-samples_per_iteration:]),
                                 append_samples=True)

    def _convert_qoi_tolist(self):
        qoi = [None] * len(self.runmodel_object.qoi_list)
        if type(self.runmodel_object.qoi_list[0]) is dict:
            for j in range(len(self.runmodel_object.qoi_list)):
                qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
        else:
            qoi = self.runmodel_object.qoi_list
        return qoi

    def _estimate_gradient(self, points, values, prediction_points):
        if self.surrogate is not None:
            self.surrogate.fit(points, values)
            self.surrogate.optimizations_number = 1
            prediction_function = self.surrogate.predict
        else:
            from scipy.interpolate import LinearNDInterpolator
            prediction_function = LinearNDInterpolator(points, values, fill_value=0).__call__

        gradient_values = gradient(point=prediction_points,
                                   runmodel_object=prediction_function,
                                   order='first', df_step=self.step_size)
        return gradient_values