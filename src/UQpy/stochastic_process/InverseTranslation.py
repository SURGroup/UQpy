from UQpy.distributions import *
from UQpy.utilities import *
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,
    wiener_khinchin_transform,
)


class InverseTranslation:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes.

    **Input:**

    * **distributions** (`list or numpy.ndarray`):
        An instance of the ``UQpy`` ``Distributions`` class defining the marginal distribution of the non-Gaussian
        stochastic process.

    * **time_interval** (`float`):
        The value of time discretization.

    * **frequency_interval** (`float`):
        The value of frequency discretization.

    * **number_time_intervals** (`int`):
        The number of time discretizations.

    * **number_frequency_intervals** (`int`):
        The number of frequency discretizations.

    * **power_spectrum_non_gaussian** ('list or numpy.ndarray'):
        The power spectrum of the non-Gaussian stochastic processes.

    * **correlation_function_non_gaussian** ('list or numpy.ndarray'):
        The auto correlation function of the non-Gaussian stochastic processes.

        Either the power spectrum or the auto correlation function of the Gaussian stochastic process needs to be
        defined.

    * **samples_non_gaussian** (`list or numpy.ndarray`):
        Samples of non-Gaussian stochastic processes.

        `samples_non_gaussian` is optional. If no samples are passed, the ``InverseTranslation`` class will compute the
        underlying Gaussian correlation using the ITAM.

    **Attributes:**

    * **samples_gaussian** (`numpy.ndarray`):
        The inverse translated Gaussian samples from the non-Gaussian samples.

    * **power_spectrum_gaussian** (`numpy.ndarray`):
        The power spectrum of the inverse translated Gaussian stochastic processes.

    * **correlation_function_gaussian** (`numpy.ndarray`):
        The correlation function of the inverse translated Gaussian stochastic processes.

    * **scaled_correlation_function_non_gaussian** (`numpy.ndarray`):
        This obtained by scaling the correlation function of the Gaussian stochastic processes to make the correlation
        at '0' distance to be 1

    """

    def __init__(
        self,
        distributions,
        time_interval,
        frequency_interval,
        number_time_intervals,
        number_frequency_intervals,
        correlation_function_non_gaussian=None,
        power_spectrum_non_gaussian=None,
        samples_non_gaussian=None,
        percentage_error=5.0
    ):
        self.distributions = distributions
        self.frequency = np.arange(0, number_frequency_intervals) * frequency_interval
        self.time = np.arange(0, number_time_intervals) * time_interval
        self.error = percentage_error
        if (
            correlation_function_non_gaussian is None
            and power_spectrum_non_gaussian is None
        ):
            print(
                "Either the Power Spectrum or the Autocorrelation function should be specified"
            )
        if correlation_function_non_gaussian is None:
            self.power_spectrum_non_gaussian = power_spectrum_non_gaussian
            self.correlation_function_non_gaussian = wiener_khinchin_transform(
                power_spectrum_non_gaussian, self.frequency, self.time
            )
        elif power_spectrum_non_gaussian is None:
            self.correlation_function_non_gaussian = correlation_function_non_gaussian
            self.power_spectrum_non_gaussian = inverse_wiener_khinchin_transform(
                correlation_function_non_gaussian, self.frequency, self.time
            )
        self.num = self.correlation_function_non_gaussian.shape[0]
        self.dim = len(self.correlation_function_non_gaussian.shape)
        if samples_non_gaussian is not None:
            self.samples_shape = samples_non_gaussian.shape
            self.samples_non_gaussian = samples_non_gaussian.flatten()[:, np.newaxis]
            self.samples_gaussian = self._inverse_translate_non_gaussian_samples().reshape(
                self.samples_shape
            )
        self.power_spectrum_gaussian = self._itam_power_spectrum()
        self.auto_correlation_function_gaussian = wiener_khinchin_transform(
            self.power_spectrum_gaussian, self.frequency, self.time
        )
        self.correlation_function_gaussian = (
            self.auto_correlation_function_gaussian
            / self.auto_correlation_function_gaussian[0]
        )

    def _inverse_translate_non_gaussian_samples(self):
        if hasattr(self.distributions, "cdf"):
            non_gaussian_cdf = getattr(self.distributions, "cdf")
            samples_cdf = non_gaussian_cdf(self.samples_non_gaussian)
        else:
            raise AttributeError(
                "UQpy: The marginal dist_object needs to have an inverse cdf defined."
            )
        samples_g = Normal(loc=0.0, scale=1.0).icdf(samples_cdf)
        return samples_g

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
