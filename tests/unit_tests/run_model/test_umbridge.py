import pytest
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.UmBridgeModel import UmBridgeModel


def test_umbridge():
    model = UmBridgeModel()
    sample = [[17.4]]
    runmodel = RunModel(model=model)

    runmodel.run(samples=sample)
    a=1