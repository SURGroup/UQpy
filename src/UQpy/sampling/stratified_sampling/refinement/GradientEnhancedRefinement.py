from beartype import beartype
from sklearn.gaussian_process import GaussianProcessRegressor

from UQpy.surrogates.baseclass import Surrogate
from UQpy.utilities.ValidationTypes import *
from UQpy.run_model.RunModel import RunModel
from UQpy.sampling.stratified_sampling.refinement.baseclass.Refinement import *
from UQpy.utilities.Utilities import gradient
from UQpy.sampling.stratified_sampling.strata.VoronoiStrata import VoronoiStrata
from UQpy.sampling.stratified_sampling.strata.baseclass.Strata import Strata

CompatibleSurrogate = Annotated[object, Is[lambda x: hasattr(x, "fit") and hasattr(x, 'predict')]]


class GradientEnhancedRefinement(Refinement):
    @beartype
    def __init__(
        self,
        strata: Strata,
        runmodel_object: RunModel,
        surrogate: Union[Surrogate, GaussianProcessRegressor, CompatibleSurrogate],
        nearest_points_number: int = None,
        qoi_name: str = None,
        step_size: float = 0.005,
    ):
        """
        Gradient-enhanced version (so-called GE-RSS) refinement algorithm. Draws samples in strata that possess both
        large probability weight and have high variance.

        :param strata: :class:`.Strata` object containing already stratified domain to be adaptively sampled using
         :class:`.RefinedStratifiedSampling`
        :param runmodel_object: A :class:`.RunModel` object, which is used to evaluate the model. It is used to compute
         the gradient of the model in each stratum.
        :param surrogate: An object defining a surrogate model.This object must have the :py:meth:`fit` and
         :py:meth:`predict` methods. This parameter aids in computing the gradient.
        :param nearest_points_number: Specifies the number of nearest points at which to update the gradient.
        :param qoi_name: Name of the quantity of interest from the runmodel_object. If the quantity of interest is a
         dictionary, this used to convert it to a list.
        :param step_size: Defines the size of the step to use for gradient estimation using the central difference
         method.
        """
        self.runmodel_object = runmodel_object
        self.step_size = step_size
        self.nearest_points_number = nearest_points_number
        self.qoi_name = qoi_name
        self.strata = strata
        self.dy_dx = 0
        if surrogate is not None:
            if hasattr(surrogate, 'fit') and hasattr(surrogate, 'predict'):
                self.surrogate = surrogate
            else:
                raise NotImplementedError("UQpy Error: surrogate must have 'fit' and 'predict' methods.")

    def update_strata(self, samplesU01):
        if isinstance(self.strata, VoronoiStrata):
            self.strata = VoronoiStrata(seeds=samplesU01)

    def initialize(self, samples_number, training_points, samples):
        self.runmodel_object.run(samples)
        self.dy_dx = np.zeros((samples_number, np.size(training_points[1])))
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

        self.strata.estimate_gradient(
            self.surrogate,
            self.step_size,
            nsamples,
            index,
            samples_u01,
            training_points,
            self._convert_qoi_tolist(),
        )

        strata_metrics = self.strata.calculate_gradient_strata_metrics(index)

        bins2break = self.identify_bins(
            strata_metrics=strata_metrics,
            points_to_add=points_to_add,
            random_state=random_state)

        new_points = self.strata.update_strata_and_generate_samples(
            dimension, points_to_add, bins2break, samples_u01, random_state)

        return new_points

    def finalize(self, samples, samples_per_iteration):
        self.runmodel_object.run(samples=np.atleast_2d(samples[-samples_per_iteration:]), append_samples=True)

    def _convert_qoi_tolist(self):
        qoi = [None] * len(self.runmodel_object.qoi_list)
        if type(self.runmodel_object.qoi_list[0]) is dict:
            for j in range(len(self.runmodel_object.qoi_list)):
                qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
        else:
            qoi = self.runmodel_object.qoi_list
        return qoi

    def _estimate_gradient(self, points, values, prediction_points):
        """
        Estimating gradients with a Kriging metamodel (surrogate).
        **Inputs:**
        * **x** (`ndarray`):
            Samples in the training data.
        * **y** (`ndarray`):
            Function values evaluated at the samples in the training data.
        * **xt** (`ndarray`):
            Samples where gradients need to be evaluated.
        **Outputs:**
        * **gr** (`ndarray`):
            First-order gradient evaluated at the points 'xt' using central difference.
        """
        if self.surrogate is not None:
            self.surrogate.fit(prediction_points, values)
            self.surrogate.optimizations_number = 1
            prediction_function = self.surrogate.predict
        else:
            from scipy.interpolate import LinearNDInterpolator

            prediction_function = LinearNDInterpolator(
                prediction_points, values, fill_value=0
            ).__call__

        gradient_values = gradient(
            point=prediction_points,
            runmodel_object=prediction_function,
            order="first",
            df_step=self.step_size,
        )
        return gradient_values
