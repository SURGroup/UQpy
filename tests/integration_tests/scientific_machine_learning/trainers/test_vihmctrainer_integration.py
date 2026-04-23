import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import UQpy.scientific_machine_learning as sml
import hamiltorch.util as util

torch.manual_seed(0)


class TestVIHMCTrainer:
    """Test the __init__ and run methods of the VIHMCTrainer

    Note:
        This test does *not* check if the trained model is accurate
    """

    x = torch.tensor([-1.0, 1.0])
    y = x ** 2
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, test_dataset = random_split(dataset, [1, 1])
    train_data = DataLoader(train_dataset)
    test_data = DataLoader(test_dataset)

    vi_model = sml.FeedForwardNeuralNetwork(sml.BayesianLinear(1, 1))
    optimizer = torch.optim.Adam(vi_model.parameters())
    epochs = 2
    vi_trainer = sml.BBBTrainer(vi_model, optimizer)
    vi_trainer.run(train_data, test_data, epochs=epochs)

    det_model = nn.Linear(1, 1)
    vihmc_trainer = sml.VIHMCTrainer(det_model, vi_model)
    num_samples = 10
    params_hmc, _, _ = vihmc_trainer.run(train_data, test_data, num_samples=num_samples)

    def test_init(self):
        """
        Checks if the prediction with mean parameters of VI, predictions using the deterministic model at VI means, and
        the functional model at VI means matches.

        """
        params_unflat = util.unflatten(self.det_model, self.vihmc_trainer.mean_params)
        self.vihmc_trainer.vi_model.sample(False)
        x = torch.Tensor([[-1], [1]])
        y_vi = self.vihmc_trainer.vi_model(x)
        for i, param in enumerate(self.vihmc_trainer.model.parameters()):
            param.data = params_unflat[i]
        y_det = self.vihmc_trainer.model(x)
        params_named_list = [n for n, _ in self.vihmc_trainer.model.named_parameters()]
        params_dict = dict(zip(params_named_list, params_unflat))
        y_func = self.vihmc_trainer.functional_model(params_dict, [x])
        assert torch.allclose(y_vi, y_func), "VI not equal to func"
        assert torch.allclose(y_vi, y_det), "VI not equal to det"

    def test_num_params(self):
        """Passes if parameters sampled from HMC has the same length as number of samples"""
        assert len(self.params_hmc) == self.num_samples
        contains_nan = any(torch.isnan(torch.tensor(self.params_hmc)))
        assert not contains_nan
