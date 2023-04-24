import numpy as np
import UQpy
from UQpy.surrogates import polynomial_chaos
from scipy.spatial.distance import cdist
from beartype import beartype


class ThetaCriterionPCE:
    @beartype
    def __init__(self, surrogates: list[UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion]):
        """
        Active learning for polynomial chaos expansion using Theta criterion balancing between exploration and
        exploitation.
        
        :param surrogates: list of objects of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class 
        """

        self.surrogates = surrogates

    def run(self, existing_samples: np.ndarray, candidate_samples: np.ndarray, nsamples=1, samples_weights=None,
            candidate_weights=None, pce_weights=None, enable_criterium: bool=False):

        """
        Execute the :class:`.ThetaCriterionPCE` active learning.

        :param existing_samples: Samples in existing ED used for construction of PCEs.
        :param candidate_samples: Candidate samples for selecting by Theta criterion.
        :param samples_weights: Weights associated to X samples (e.g. from Coherence Sampling).
        :param candidate_weights: Weights associated to candidate samples (e.g. from Coherence Sampling).
        :param nsamples: Number of samples selected from candidate set in a single run of this algorithm
        :param pce_weights: Weights associated to each PCE (e.g. Eigen values from dimension-reduction techniques)
        :param enable_criterium: If True, values of Theta criterion (variance density, average variance density, geometrical part, total Theta criterion) for all
         candidates are returned instead of a positions of best candidates
         The :meth:`run` method is the function that performs iterations in the :class:`.ThetaCriterionPCE` class.
         The :meth:`run` method of the :class:`.ThetaCriterionPCE` class can be invoked many times for sequential
         sampling.
        :return: Position of the best candidate in candidate set. If ``enable_criterium = True``, values of Theta
         criterion (variance density, average variance density, geometrical part, total Theta criterion) for all
         candidates are returned instead of a position.
        """

        pces = self.surrogates

        npce = len(pces)
        nsimexisting, nvar = existing_samples.shape
        nsimcandidate, nvar = candidate_samples.shape
        criterium = np.zeros(nsimcandidate)
        if samples_weights is None:
            samples_weights = np.ones(nsimexisting)

        if candidate_weights is None:
            candidate_weights = np.ones(nsimcandidate)

        if pce_weights is None:
            pce_weights = np.ones(npce)

        pos = []

        for _ in range(nsamples):
            S = polynomial_chaos.Polynomials.standardize_sample(existing_samples, pces[0].polynomial_basis.distributions)
            s_candidate = polynomial_chaos.Polynomials.standardize_sample(candidate_samples,
                                                                         pces[0].polynomial_basis.distributions)

            lengths = cdist(s_candidate, S)
            closest_s_position = np.argmin(lengths, axis=1)
            closest_value_x = existing_samples[closest_s_position]
            l = np.nanmin(lengths, axis=1)
            variance_candidate = 0
            variance_closest = 0

            for i in range(npce):
                pce = pces[i]
                variance_candidatei = self._local_variance(candidate_samples, pce, candidate_weights)
                variance_closesti = self._local_variance(closest_value_x, pce, samples_weights[closest_s_position])

                variance_candidate = variance_candidate + variance_candidatei * pce_weights[i]
                variance_closest = variance_closest + variance_closesti * pce_weights[i]

            criterium_v = np.sqrt(variance_candidate * variance_closest)
            criterium_l = l ** nvar
            criterium = criterium_v * criterium_l
            pos.append(np.argmax(criterium))
            existing_samples = np.append(existing_samples, candidate_samples[pos, :], axis=0)
            samples_weights = np.append(samples_weights, candidate_weights[pos])

        if not enable_criterium:
            if nsamples == 1:
                pos = pos[0]
            return pos
        else:
            return variance_candidate, criterium_v, criterium_l, criterium

    # calculate variance density of PCE for Theta Criterion
    @staticmethod
    def _local_variance(coordinates, pce, weight=1):
        beta = pce.coefficients
        beta[0] = 0

        product = pce.polynomial_basis.evaluate_basis(coordinates)

        product = np.transpose(np.transpose(product) * weight)
        product = product.dot(beta)

        product = np.sum(product, axis=1)

        product = product ** 2
        product = product * polynomial_chaos.Polynomials.standardize_pdf(coordinates,
                                                                         pce.polynomial_basis.distributions)

        return product
