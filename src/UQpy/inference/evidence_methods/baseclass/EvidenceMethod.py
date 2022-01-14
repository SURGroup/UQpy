from abc import ABC, abstractmethod


class EvidenceMethod(ABC):
    @abstractmethod
    def estimate_evidence(self, inference_model, posterior_samples, log_posterior_values):
        pass
