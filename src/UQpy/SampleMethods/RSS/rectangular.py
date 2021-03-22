from UQpy.SampleMethods.RSS.rss import RSS
from UQpy.SampleMethods.STS import RectangularSTS
import numpy as np
import scipy.stats as stats
import copy


class RectangularRSS(RSS):
    """
    Executes Refined Stratified Sampling using Rectangular Stratification.

    ``RectangularRSS`` is a child class of ``RSS``. ``RectangularRSS`` takes in all parameters defined in the parent
    ``RSS`` class with differences note below. Only those inputs and attributes that differ from the parent class
    are listed below. See documentation for ``RSS`` for additional details.

    **Inputs:**

    * **sample_object** (``RectangularSTS`` object):
        The `sample_object` for ``RectangularRSS`` must be an object of the ``RectangularSTS`` class.

    **Methods:**
    """
    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        if not isinstance(sample_object, RectangularSTS):
            raise NotImplementedError("UQpy Error: sample_object must be an object of the RectangularSTS class.")

        self.strata_object = copy.deepcopy(sample_object.strata_object)

        super().__init__(sample_object=sample_object, runmodel_object=runmodel_object, krig_object=krig_object,
                         local=local, max_train_size=max_train_size, step_size=step_size, qoi_name=qoi_name,
                         n_add=n_add, nsamples=nsamples, random_state=random_state, verbose=verbose)

    def run_rss(self):
        """
        Overwrites the ``run_rss`` method in the parent class to perform refined stratified sampling with rectangular
        strata. It is an instance method that does not take any additional input arguments. See
        the ``RSS`` class for additional details.
        """
        if self.runmodel_object is not None:
            self._gerss()
        else:
            self._rss()

        self.weights = self.strata_object.volume

    def _gerss(self):
        """
        This method generates samples using Gradient Enhanced Refined Stratified Sampling.
        """
        if self.verbose:
            print('UQpy: Performing GE-RSS with rectangular stratification...')

        # Initialize the vector of gradients at each training point
        dy_dx = np.zeros((self.nsamples, np.size(self.training_points[1])))

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                qoi = self.runmodel_object.qoi_list

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the gradients at the existing sample points
            if self.max_train_size is None or len(
                    self.training_points) <= self.max_train_size or i == self.samples.shape[0]:
                # Use the entire sample set to train the surrogate model (more expensive option)
                dy_dx[:i] = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                   np.atleast_2d(np.array(qoi)),
                                                   self.strata_object.seeds +
                                                   0.5 * self.strata_object.widths)
            else:
                # Use only max_train_size points to train the surrogate model (more economical option)
                # Find the nearest neighbors to the most recently added point
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=self.max_train_size)
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

            # ###########################
            # ---------------------------
            # 3. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###############################
            # -------------------------------
            # 4. Execute model at new samples
            # -------------------------------
            self.runmodel_object.run(samples=np.atleast_2d(self.samples[-self.n_add:]), append_samples=True)

            if self.verbose:
                print("Iteration:", i)

    def _rss(self):
        """
        This method generates samples using Refined Stratified Sampling.
        """

        if self.verbose:
            print('UQpy: Performing RSS with rectangular stratification...')

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration
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

            # ###########################
            # ---------------------------
            # 3. Update sample attributes
            # ---------------------------
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
        new = stats.uniform.rvs(loc=self.strata_object.seeds[-1, :], scale=self.strata_object.widths[-1, :],
                                random_state=self.random_state)

        return new