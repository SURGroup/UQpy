from UQpy.distributions import *
from UQpy.utilities import *
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,
    wiener_khinchin_transform,
)


class InverseTranslation:
    def __init__(
        self,
        distributions: Distribution,
        time_interval: Union[list, np.ndarray],
        frequency_interval: Union[list, np.ndarray],
        n_time_intervals: Union[list, np.ndarray],
        n_frequency_intervals: Union[list, np.ndarray],
        correlation_function_non_gaussian: Union[list, np.ndarray] = None,
        power_spectrum_non_gaussian: Union[list, np.ndarray] = None,
        samples_non_gaussian: Union[list, np.ndarray] = None,
        percentage_error: float = 5.0
    ):
        """
        A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic
        Processes which upon translation would yield the necessary non-Gaussian Stochastic Processes.

        :param distributions: An instance of the :py:mod:`UQpy` :class:`.Distributions` class defining the marginal
         distribution of the non-Gaussian stochastic process.
        :param time_interval: The value of time discretization.
        :param frequency_interval: The value of frequency discretization.
        :param n_time_intervals: The number of time discretizations.
        :param n_frequency_intervals: The number of frequency discretizations.
        :param correlation_function_non_gaussian: The auto correlation function of the non-Gaussian stochastic
         processes. Either the power spectrum or the auto correlation function of the Gaussian stochastic process needs
         to be defined.
        :param power_spectrum_non_gaussian: The power spectrum of the non-Gaussian stochastic processes.
        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes.
         `samples_non_gaussian` is optional. If no samples are passed, the :class:`.InverseTranslation` class will
         compute the underlying Gaussian correlation using the ITAM.
        :param percentage_error:
        """
        self.samples_gaussian = None
        """The inverse translated Gaussian samples from the non-Gaussian samples."""
        self.samples_non_gaussian = None
        self.samples_shape = None
        self.distributions = distributions
        self.frequency = np.arange(0, n_frequency_intervals) * frequency_interval
        self.time = np.arange(0, n_time_intervals) * time_interval
        self.error = percentage_error
        self.logger = logging.getLogger(__name__)
        if correlation_function_non_gaussian is None and power_spectrum_non_gaussian is None:
            self.logger.info("Either the Power Spectrum or the Autocorrelation function should be specified")
        if correlation_function_non_gaussian is None:
            self.power_spectrum_non_gaussian = power_spectrum_non_gaussian
            self.correlation_function_non_gaussian = wiener_khinchin_transform(power_spectrum_non_gaussian,
                                                                               self.frequency, self.time)
        elif power_spectrum_non_gaussian is None:
            self.correlation_function_non_gaussian = correlation_function_non_gaussian
            self.power_spectrum_non_gaussian = inverse_wiener_khinchin_transform(correlation_function_non_gaussian,
                                                                                 self.frequency, self.time)
        self.num = self.correlation_function_non_gaussian.shape[0]
        self.dim = len(self.correlation_function_non_gaussian.shape)

        if samples_non_gaussian is not None:
            self.run(samples_non_gaussian)

        self.power_spectrum_gaussian: NumpyFloatArray = self._itam_power_spectrum()
        """The power spectrum of the inverse translated Gaussian stochastic processes"""
        self.auto_correlation_function_gaussian = wiener_khinchin_transform(
            self.power_spectrum_gaussian, self.frequency, self.time)
        self.correlation_function_gaussian: NumpyFloatArray = (
            self.auto_correlation_function_gaussian / self.auto_correlation_function_gaussian[0])
        """The correlation function of the inverse translated Gaussian stochastic processes."""

    def run(self, samples_non_gaussian):
        """

        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes. If samples are provided at the
         initialization, then the run method is executed automatically. If no samples are passed at the
         initialization, the :class:`.InverseTranslation` class will compute the underlying Gaussian correlation using
         the ITAM and the run method needs to be manually executed by the user.
        """

        self.samples_shape = samples_non_gaussian.shape
        self.samples_non_gaussian = samples_non_gaussian.flatten()[:, np.newaxis]
        self.samples_gaussian: NumpyFloatArray = self._inverse_translate_non_gaussian_samples().reshape(
            self.samples_shape)


    def _inverse_translate_non_gaussian_samples(self):
        if not hasattr(self.distributions, "cdf"):
            raise AttributeError("UQpy: The marginal dist_object needs to have an inverse cdf defined.")
        non_gaussian_cdf = getattr(self.distributions, "cdf")
        samples_cdf = non_gaussian_cdf(self.samples_non_gaussian)
        return Normal(loc=0.0, scale=1.0).icdf(samples_cdf)

    def _itam_power_spectrum(self):
        target_S = self.power_spectrum_non_gaussian
        i_converge = 0
        max_iter = 500
        target_R = wiener_khinchin_transform(target_S, self.frequency, self.time)
        R_g_iterate = target_R
        S_g_iterate = target_S
        R_ng_iterate = np.zeros_like(R_g_iterate)
        r_ng_iterate = np.zeros_like(R_g_iterate)
        S_ng_iterate = np.zeros_like(S_g_iterate)
        non_gaussian_moments = getattr(self.distributions, 'moments')()

        for _ in range(max_iter):
            R_g_iterate = wiener_khinchin_transform(S_g_iterate, self.frequency, self.time)
            for i in range(len(target_R)):
                r_ng_iterate[i] = correlation_distortion(dist_object=self.distributions,
                                                         rho=R_g_iterate[i] / R_g_iterate[0])
            R_ng_iterate = r_ng_iterate * non_gaussian_moments[1] + non_gaussian_moments[0] ** 2
            S_ng_iterate = inverse_wiener_khinchin_transform(R_ng_iterate, self.frequency, self.time)

            err1 = np.sum((target_S - S_ng_iterate) ** 2)
            err2 = np.sum(target_S ** 2)

            if 100 * np.sqrt(err1 / err2) < self.error:
                i_converge = 1

            ratio = target_S / S_ng_iterate
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

            S_g_next_iterate = (ratio ** 1.3) * S_g_iterate

            # Eliminate Numerical error of Upgrading Scheme
            S_g_next_iterate[S_g_next_iterate < 0] = 0
            S_g_iterate = S_g_next_iterate

            if i_converge:
                break

        return S_g_iterate
