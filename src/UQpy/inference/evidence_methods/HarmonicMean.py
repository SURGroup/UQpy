from UQpy.inference.evidence_methods.baseclass.EvidenceMethod import EvidenceMethod
import numpy as np


class HarmonicMean(EvidenceMethod):
    """
    Class used for the computation of model evidence using the harmonic mean method.
    """
    def estimate_evidence(self, inference_model, posterior_samples, log_posterior_values):
        log_likelihood_values = (log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples))
        temp = np.mean(1.0 / np.exp(log_likelihood_values))
        return 1.0 / temp
