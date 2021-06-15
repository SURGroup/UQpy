from UQpy.sample_methods.refined_stratified.RefinedStratifiedSampling import RefinedStratifiedSampling
from UQpy.sample_methods.stratifications import RectangularSTS
import numpy as np
import scipy.stats as stats
import copy


class RectangularRefinedStratifiedSampling(RefinedStratifiedSampling):
    """
    Executes Refined Stratified Sampling using Rectangular Stratification.

    ``RectangularRSS`` is a child class of ``refined_stratified``. ``RectangularRSS`` takes in all parameters defined
      in the parent ``refined_stratified`` class with differences note below. Only those inputs and attributes that
      differ from the parent class are listed below. See documentation for ``refined_stratified`` for additional details

    **Inputs:**

    * **sample_object** (``RectangularSTS`` object):
        The `sample_object` for ``RectangularRSS`` must be an object of the ``RectangularSTS`` class.

    **Methods:**
    """
    def __init__(self, sample_object=None, runmodel_object=None, kriging=None, update_locally=False,
                 nearest_points_number=None, step_size=0.005, qoi_name=None, new_iteration_samples=1,
                 samples_number=None, random_state=None, verbose=False):

        if not isinstance(sample_object, RectangularSTS):
            raise NotImplementedError("UQpy Error: sample_object must be an object of the RectangularSTS class.")

        self.strata_object = copy.deepcopy(sample_object.strata_object)

        super().__init__(sample_object=sample_object, runmodel_object=runmodel_object, kriging=kriging,
                         update_locally=update_locally, nearest_points_number=nearest_points_number,
                         step_size=step_size, qoi_name=qoi_name, new_iteration_samples=new_iteration_samples,
                         samples_number=samples_number, random_state=random_state, verbose=verbose)

    def run_refined_stratified_sampling(self):
        """
        Overwrites the ``run_rss`` method in the parent class to perform refined stratified sampling with rectangular
        strata. It is an instance method that does not take any additional input arguments. See
        the ``refined_stratified`` class for additional details.
        """
        if self.runmodel_object is not None:
            self._generate_gradient_enhanced_samples()
        else:
            self._generate_samples()

        self.weights = self.strata_object.volume

    def _generate_gradient_enhanced_samples(self):
        """
        This method generates samples using Gradient Enhanced Refined Stratified Sampling.
        """
        if self.verbose:
            print('UQpy: Performing GE-refined_stratified with rectangular stratification...')

        # Initialize the vector of gradients at each training point
        dy_dx = np.zeros((self.samples_number, np.size(self.training_points[1])))

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.samples_number, self.new_iteration_samples):
            p = min(self.new_iteration_samples, self.samples_number - i)  # Number of points to add in this iteration

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = self._convert_qoi_tolist()

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the gradients at the existing sample points
            if self.nearest_points_number is None or len(
                    self.training_points) <= self.nearest_points_number or i == self.samples.shape[0]:
                # Use the entire sample set to train the surrogate model (more expensive option)
                dy_dx[:i] = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                   np.atleast_2d(np.array(qoi)),
                                                   self.strata_object.seeds +
                                                   0.5 * self.strata_object.widths)
            else:
                # Use only max_train_size points to train the surrogate model (more economical option)
                # Find the nearest neighbors to the most recently added point
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=self.nearest_points_number)
                knn.fit(np.atleast_2d(self.training_points))
                neighbors = knn.kneighbors(np.atleast_2d(self.training_points[-1]), return_distance=False)

                # Recompute the gradient only at the nearest neighbor points.
                dy_dx[neighbors] = self.estimate_gradient(np.squeeze(self.training_points[neighbors]),
                                                          np.array(qoi)[neighbors][0],
                                                          np.squeeze(
                                                              self.strata_object.seeds[neighbors] +
                                                              0.5 * self.strata_object.widths[
                                                                  neighbors]))

            # Define the gradient vector for application of the Delta Method
            dy_dx1 = dy_dx[:i]

            # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
            # All input variables are independent
            var = (1 / 12) * self.strata_object.widths ** 2

            # Estimate the variance over the stratum by Delta Method
            s = np.zeros([i])
            for j in range(i):
                s[j] = np.sum(dy_dx1[j, :] * var[j, :] * dy_dx1[j, :]) * self.strata_object.volume[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2break = self.identify_bins(strata_metric=s, p_=p)

            # #############################################
            # ---------------------------------------------
            # 2. Update each strata and generate new sample
            # ---------------------------------------------
            new_points = np.zeros([p, self.dimension])
            # Update the strata_object for all new points
            for j in range(p):
                new_points[j, :] = self._update_stratum_and_generate_sample(bin2break[j])

            self.update_samples(new_point=new_points)

            self.runmodel_object.run(samples=np.atleast_2d(self.samples[-self.new_iteration_samples:]),
                                     append_samples=True)

            if self.verbose:
                print("Iteration:", i)

    def _generate_samples(self):
        """
        This method generates samples using Refined Stratified Sampling.
        """

        if self.verbose:
            print('UQpy: Performing refined_stratified with rectangular stratification...')

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.samples_number, self.new_iteration_samples):
            p = min(self.new_iteration_samples, self.samples_number - i)  # Number of points to add in this iteration
            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------
            # Estimate the weight corresponding to each stratum
            s = np.zeros(i)
            for j in range(i):
                s[j] = self.strata_object.volume[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2break = self.identify_bins(strata_metric=s, p_=p)

            # #############################################
            # ---------------------------------------------
            # 2. Update each strata and generate new sample
            # ---------------------------------------------
            new_points = np.zeros([p, self.dimension])
            # Update the strata_object for all new points, 'p' is number of samples to be added in the current iteration
            for j in range(p):
                new_points[j, :] = self._update_stratum_and_generate_sample(bin2break[j])

            self.update_samples(new_point=new_points)

            if self.verbose:
                print("Iteration:", i)

    def _update_stratum_and_generate_sample(self, bin_):
        # Cut the stratum in the direction of maximum length
        cut_dir_temp = self.strata_object.widths[bin_, :]
        dir2break = np.random.choice(np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))[0])

        # Divide the stratum bin2break in the direction dir2break
        self.strata_object.widths[bin_, dir2break] = self.strata_object.widths[bin_, dir2break] / 2
        self.strata_object.widths = np.vstack([self.strata_object.widths, self.strata_object.widths[bin_, :]])
        self.strata_object.seeds = np.vstack([self.strata_object.seeds, self.strata_object.seeds[bin_, :]])
        # print(self.samplesU01[bin_, dir2break], self.strata_object.seeds[bin_, dir2break] + \
        #       self.strata_object.widths[bin_, dir2break])
        if self.samplesU01[bin_, dir2break] < self.strata_object.seeds[bin_, dir2break] + \
                self.strata_object.widths[bin_, dir2break]:
            self.strata_object.seeds[-1, dir2break] = self.strata_object.seeds[bin_, dir2break] + \
                                                      self.strata_object.widths[bin_, dir2break]
            # print("retain")
        else:
            self.strata_object.seeds[bin_, dir2break] = self.strata_object.seeds[bin_, dir2break] + \
                                                        self.strata_object.widths[bin_, dir2break]

        self.strata_object.volume[bin_] = self.strata_object.volume[bin_] / 2
        self.strata_object.volume = np.append(self.strata_object.volume, self.strata_object.volume[bin_])

        # Add a uniform random sample inside the new stratum
        new_samples = stats.uniform.rvs(loc=self.strata_object.seeds[-1, :], scale=self.strata_object.widths[-1, :],
                                        random_state=self.random_state)

        return new_samples
