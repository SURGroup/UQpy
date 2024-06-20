import logging
import numpy as np
from scipy import integrate
from UQpy.distributions import Distribution, Normal, MultivariateNormal
from UQpy.utilities.ValidationTypes import PositiveFloat, PositiveInteger
from typing import Union


class InverseTranslationNew:
    def __init__(
        self,
        distribution: Distribution,
        time_interval: PositiveFloat,
        frequency_interval: PositiveFloat,
        n_time_intervals: PositiveInteger,
        n_frequency_intervals: PositiveInteger,
        target_s_ng: np.ndarray = None,
        target_r_ng: np.ndarray = None,
        samples_ng: np.ndarray = None,
        threshold: PositiveFloat = 0.05,
        max_iter: PositiveInteger = 100,
    ):
        """Hellow orld

        :param distribution:
        :param time_interval:
        :param frequency_interval:
        :param n_time_intervals:
        :param n_frequency_intervals:
        :param target_s_ng:
        :param target_r_ng:
        :param samples_ng:
        :param threshold:
        """
        self.distribution = distribution
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.n_time_intervals = n_time_intervals
        self.n_frequency_intervals = n_frequency_intervals
        self.target_s_ng = (
            target_s_ng if (target_s_ng is not None) else np.fft.rfft(target_r_ng)
        )
        self.target_r_ng = (
            target_r_ng if (target_r_ng is not None) else np.fft.irfft(target_s_ng)
        )
        self.samples_ng = samples_ng
        self.threshold = threshold
        self.max_iter = max_iter

        self.logger = logging.getLogger("UQpy")
        variance_ng = self.distribution.moments(moments2return="v")
        self.normal = Normal(scale=float(variance_ng))

        s_g, s_ng = self.compute_itam_power_spectrum()
        self.s_g = s_g
        self.s_ng = s_ng
        self.r_g = np.fft.fft(self.s_g)

    def correlation_distortion_integrand(self, x1, x2, var, rho):
        """Compute the correlation distortion integrand as defined by equation 6 from Shields 2011

        :param x1:
        :param x2:
        :param cov:
        :return:
        """
        constant = 1 / (2 * np.pi * var * np.sqrt(1 - rho**2))
        power = -(x1**2 + x2**2 - (2 * rho * x1 * x2)) / (2 * var * (1 - rho**2))
        bivariate_normal_pdf = constant * np.exp(power)
        # bivariate_input = np.array([x1, x2])
        # bivariate_normal = MultivariateNormal(mean=[0.0, 0.0], cov=cov)
        return (
            self.distribution.icdf(self.normal.cdf(x1))
            * self.distribution.icdf(self.normal.cdf(x2))
            * bivariate_normal_pdf
        )

    def compute_itam_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Gaussian and reconstructed non-Gaussian power spectrum from the target non-Gaussian power spectrum

        :return: Gaussian power spectrum, reconstructed non-Gaussian power spectrum
        """
        lower_bound = -1
        upper_bound = 1
        s_g = self.target_s_ng.copy()
        s_ng = np.zeros_like(s_g)
        r_g = self.target_r_ng.copy()
        r_ng = np.zeros_like(r_g)
        mean_ng, variance_ng = self.distribution.moments(moments2return="mv")
        mean_ng = float(mean_ng)
        variance_ng = float(variance_ng)
        i = 0
        error = np.inf
        self.logger.info(
            f"UQpy: Stochastic Process: Iteration {i:,} / {self.max_iter:,} Error {error:.6e}"
        )
        while (i < self.max_iter) and (error > self.threshold):
            r_g = np.real(np.fft.irfft(s_g))
            # print(i, r_g)
            variance_g = r_g[0]
            print(i, variance_g)
            # assert (
            #     r_g[0] > abs(r_g[1:])
            # ).all(), "Autocorrelation R_g(0) is not the maximum"
            if variance_g == 0:
                raise ValueError(f"UQpy: Iteration {i} computed r_g[0] == 0")
            for j in range(len(r_g)):
                if j == 0:
                    rho = 0
                else:
                    rho = r_g[j] / variance_g
                # cov = np.array([[variance_g, rho], [rho, variance_g]])
                r_ng[j], _ = integrate.dblquad(
                    lambda x1, x2: self.correlation_distortion_integrand(
                        x1, x2, variance_g, rho
                    ),
                    lower_bound,
                    upper_bound,
                    lower_bound,
                    upper_bound,
                )
            r_ng = (r_ng * variance_ng) + (mean_ng**2)
            s_ng = np.real(np.fft.rfft(r_ng))
            base = self.target_s_ng / s_ng
            base = np.nan_to_num(
                base, nan=0.0, posinf=0.0, neginf=0.0
            )  # Set NaN, +inf, -inf to 0
            base = np.clip(base, 0, None)  # Set negatives to 0
            s_g = (base**1.3) * s_g
            error = np.sum((s_ng - self.target_s_ng) ** 2) / np.sum(self.target_s_ng**2)
            error = np.sqrt(error)
            i += 1
            self.logger.info(
                f"UQpy: Stochastic Process: Iteration {i:,} / {self.max_iter:,} Error {error:.6e}"
            )
        self.logger.info(
            f"UQpy: Stochastic Process: Ended Inverse Translation Approximation Method"
        )
        if error > self.threshold:
            self.logger.warning(
                "UQpy: Stochastic Process: InverseTranslation may have undesirably large error"
            )
        return s_g, s_ng


if __name__ == "__main__":
    import UQpy.distributions.collection as dist

    logger = logging.getLogger("UQpy")
    logger.setLevel(logging.INFO)

    n_frequency_interval = 64
    max_frequency = np.pi  # based on figure 4
    frequency_interval = max_frequency / n_frequency_interval
    n_time_interval = 128
    max_time = 256  # based on figure 5
    time_interval = max_time / n_time_interval

    frequency = np.linspace(
        0, (n_frequency_interval - 1) * frequency_interval, num=n_frequency_interval
    )
    target_s_ng = (125 / 4) * (frequency**2) * np.exp(-5 * abs(frequency))
    distribution = dist.Lognormal(s=1, loc=-1.8)

    itam = InverseTranslationNew(
        distribution,
        time_interval,
        frequency_interval,
        n_time_interval,
        n_frequency_interval,
        target_s_ng,
    )
