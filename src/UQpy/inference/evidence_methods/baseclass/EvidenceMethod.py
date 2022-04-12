from abc import ABC, abstractmethod

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.inference.inference_models.baseclass import InferenceModel


class EvidenceMethod(ABC):
    @abstractmethod
    def estimate_evidence(self, inference_model: InferenceModel,
                          posterior_samples: NumpyFloatArray,
                          log_posterior_values: NumpyFloatArray) -> float:
        """

        :param inference_model: Probabilistic model used for inference.
        :param posterior_samples: Samples drawn from the posterior distribution of the parameters using a
         :class:`.BayesParameterEstimation` object.
        :param log_posterior_values: Values of the ``log_pdf`` function generated during the sampling of the
         :class:`.BayesParameterEstimation` object.
        :return: The evidence of the inference specific model.
        """
        pass
