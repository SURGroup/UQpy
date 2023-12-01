from abc import ABC, abstractmethod


class ComputationalModelType(ABC):
    @abstractmethod
    def initialize(self, samples):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def preprocess_single_sample(self, i, sample):
        pass

    @abstractmethod
    def execute_single_sample(self, index, sample_to_send):
        pass

    @abstractmethod
    def postprocess_single_file(self, index, model_output):
        pass
