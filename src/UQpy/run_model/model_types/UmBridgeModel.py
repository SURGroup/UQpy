import umbridge
import logging
from beartype import beartype
import importlib
import numpy as np
from UQpy.run_model.model_types import ComputationalModelType

spec = importlib.util.find_spec('umbridge')
if spec is None:
    raise ImportError("UM-Bridge library is not installed, use the command:\n"
                      "pip install umbridge\n"
                      "to enable this functionality.")
    print('module is installed')


class UmBridgeModel(ComputationalModelType):

    @beartype
    def __init__(self, url: str = 'http://localhost:4242',
                 var_names: list[str] = None,
                 **model_object_name_kwargs):
        if var_names is None:
            var_names = []
        self.var_names = var_names
        self._model_output = None
        self.logger = logging.getLogger(__name__)
        self.model_object_name_kwargs = model_object_name_kwargs

        self.logger.info(f"Model supported by server: {umbridge.supported_models(url)}")
        self.umbridge_model = umbridge.HTTPModel(url, "forward")

    def initialize(self, samples):
        pass

    def finalize(self):
        pass

    def preprocess_single_sample(self, i, sample):
        return [sample.tolist()]

    def execute_single_sample(self, index, sample_to_send):
        if len(self.model_object_name_kwargs) == 0:
            return self.umbridge_model(sample_to_send)
        else:
            return self.umbridge_model(sample_to_send, **self.model_object_name_kwargs)

    def postprocess_single_file(self, index, model_output):
        return np.array(model_output).squeeze()
