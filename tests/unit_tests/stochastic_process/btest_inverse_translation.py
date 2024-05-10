"""
The following tests for the InverseTranslation class are based on Section 7 Numerical Examples of Shields 2011

Reference
---------

**Shields 2011**
M.D. Shields, G. Deodatis, P. Bocchini,
A simple and efficient methodology to approximate a general non-Gaussian stationary stochastic process by a translation process,
Probabilistic Engineering Mechanics, Volume 26, Issue 4, 2011, Pages 511-519,
ISSN 0266-8920, https://doi.org/10.1016/j.probengmech.2011.04.003.
"""

import pytest
import numpy as np
from UQpy.distributions import Distribution, Beta, Normal, Lognormal, Uniform
from UQpy.stochastic_process import InverseTranslation


def exponential_autocorrelation(tau: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Exponential Autocorrelation function :math:`R(\\tau)`

    Note:
        This it the analytical autocorrelation function for ``exponential_power_spectrum``

    :param tau: Time lag
    :param a: Shape parameter
    :return: Autocorrelation function evaluated at the times ``tau``
    """
    return np.exp(-a * abs(tau))


def exponential_power_spectrum(omega: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Exponential power spectrum :math:`S(\\omega)`

    :param omega: Frequencies
    :param a: Shape parameter
    :return: Power spectrum evaluated at the frequencies ``omega``
    """
    numerator = 4 * a
    denominator = a**2 + (2 * np.pi * omega) ** 2
    return numerator / denominator


def compute_target_s_ng(omega: np.ndarray) -> np.ndarray:
    """Target non-Gaussian power spectrum :math:`S^T_{NG}(\\omega)` defined by Eq. 20 of Shields 2011

    :param omega: Frequencies
    :return: Power spectrum evaluated at frequencies ``omega``
    """
    return (125 / 4) * (omega**2) * np.exp(-5 * abs(omega))


def initialize_inverse_translation(
    distributions: Distribution,
    target_power_spectrum_non_gaussian: np.ndarray = None,
    target_correlation_function_non_gaussian: np.ndarray = None,
) -> InverseTranslation:
    """Construct the InverseTranslation object with a default time and frequency domain

    :param distributions: UQpy distributions defining Inverse Translation mapping
    :param target_power_spectrum_non_gaussian: :math:`S^T_{NG}(\\omega)`
    :param target_correlation_function_non_gaussian: :math:`R^T_{NG}(\\tau)`
    :return: InverseTranslation object
    """
    max_frequency = np.pi
    n_frequency_intervals = 128
    frequency_interval = max_frequency / n_frequency_intervals

    max_time = 2 * np.pi / frequency_interval
    n_time_intervals = 256
    time_interval = max_time / n_time_intervals

    itam = InverseTranslation(
        distributions,
        time_interval,
        frequency_interval,
        n_time_intervals,
        n_frequency_intervals,
        target_power_spectrum_non_gaussian=target_power_spectrum_non_gaussian,
        target_correlation_function_non_gaussian=target_correlation_function_non_gaussian,
    )
    return itam


def btest_neither_power_spectrum_nor_correlation_function():
    """InverseTranslation should raise RunTimeError when neither power spectrum nor correlation are defined"""
    with pytest.raises(RuntimeError):
        itam = initialize_inverse_translation(
            Uniform(),
            target_power_spectrum_non_gaussian=None,
            target_correlation_function_non_gaussian=None,
        )


def btest_both_power_spectrum_and_correlation_function():
    """InverseTranslation should raise RunTimeError when both power spectrum and correlation are defined"""
    with pytest.raises(RuntimeError):
        itam = initialize_inverse_translation(
            Uniform(),
            target_power_spectrum_non_gaussian=np.array([1]),
            target_correlation_function_non_gaussian=np.array([1]),
        )


def btest_aliasing_raises_error():
    """When the time_interval is too large InverseTranslation should raise RuntimeError to prevent aliaising"""
    with pytest.raises(RuntimeError):
        itam = InverseTranslation(
            Uniform(),
            1_000,
            np.pi / 128,
            256,
            128,
            target_power_spectrum_non_gaussian=np.array([1]),
        )


def btest_lognormal_power_spectrum():
    """
    Test the reconstructed non-Gaussian power spectrum using Lognormal distribution from Table 1 of Shields 2011.
    Table 3 of Shields 2011 says the reconstruction should be within 1% of the target
    """
    omega = np.linspace(0, np.pi, 128)
    target_power_spectrum_non_gaussian = compute_target_s_ng(omega)
    itam = initialize_inverse_translation(
        Lognormal(s=1, loc=-1.8),
        target_power_spectrum_non_gaussian=target_power_spectrum_non_gaussian,
    )
    assert np.allclose(
        itam.power_spectrum_non_gaussian, target_power_spectrum_non_gaussian, rtol=0.01
    )


def btest_uniform_power_spectrum():
    """
    Test the reconstructed non-Gaussian power spectrum using Uniform distribution from Table 1 of Shields 2011.
    Table 3 of Shields 2011 says the reconstruction should be within 1% of the target
    """
    omega = np.linspace(0, np.pi, 128)
    target_power_spectrum_non_gaussian = compute_target_s_ng(omega)
    itam = initialize_inverse_translation(
        Uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        target_power_spectrum_non_gaussian=target_power_spectrum_non_gaussian,
    )
    assert np.allclose(
        itam.power_spectrum_non_gaussian, target_power_spectrum_non_gaussian, rtol=0.01
    )


def btest_beta_power_spectrum():
    """
    Test the reconstructed non-Gaussian power spectrum using U-Shaped Beta distribution from Table 1 of Shields 2011.
    Table 3 of Shields 2011 says the reconstruction should be within 2% of the target
    """
    omega = np.linspace(0, np.pi, 128)
    target_power_spectrum_non_gaussian = compute_target_s_ng(omega)
    itam = initialize_inverse_translation(
        Beta(0.1895, 11.795, loc=-0.4457, scale=28.43),
        target_power_spectrum_non_gaussian=target_power_spectrum_non_gaussian,
    )
    assert np.allclose(
        itam.power_spectrum_non_gaussian, target_power_spectrum_non_gaussian, rtol=0.02
    )


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("UQpy")
    logger.setLevel(logging.INFO)
    # if logger.hasHandlers():
    #     logger.removeHandler(
    #         logger.handlers[0]
    #     )  # remove existing handlers to eliminate print statements
    # file_handler = logging.FileHandler("itam.log")
    # logger.addHandler(file_handler)

    # btest_beta_power_spectrum()
    # btest_lognormal_power_spectrum()

    btest_uniform_power_spectrum()
