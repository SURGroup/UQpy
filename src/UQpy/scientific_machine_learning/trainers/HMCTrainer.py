from torch.nn import functional as F
from torch import nn
import torch
import os
import hamiltorch
from beartype import beartype


@beartype
class HMCTrainer:
    def __init__(self, 
        net: nn.Module, 
        device: torch.device, 
        params_hmc_path: str):
        
        """Prepare to train a Bayesian neural network using Hamiltonina Monte Carlo

        :param model: Neural Network model to be trained
        :param device: Device to run the model on
        """
        self.net = net.to(device)
        self.device = device
        self.params_hmc_path = params_hmc_path
        self.params_init = self._initialize_params()
        
    def _initialize_params(self):
        "Initialize the parameters of the model. Load from file if available."
        if self.params_hmc_path and os.path.exists(self.params_hmc_path):
            return torch.load(self.params_hmc_path, map_location=self.device)[0]
        else:
            return hamiltorch.util.flatten(self.net).to(self.device).clone()

    def _save_params_hmc(self, params_hmc):
        torch.save(params_hmc, self.params_hmc_path)

    def run_training(self, 
                     X_tr: torch.Tensor,
                     Y_tr: torch.Tensor,
                     X_val: torch.Tensor,
                     Y_val: torch.Tensor,
                     loss_function: nn.Module = nn.MSELoss(),
                     num_samples_hmc: int = 1000,
                     L: int = 10,
                     step_size: float = 0.1,
                     burn_sample: int = 0,
                     burn_pred: int = 0,
                     tau_out: float = 1.,
                     tau_list: torch.Tensor = None,
                     store_on_GPU: bool = False,
                     sampler: hamiltorch.Sampler = hamiltorch.Sampler.HMC,
                     integrator: hamiltorch.Integrator = hamiltorch.Integrator.IMPLICIT,
                     metric: hamiltorch.Metric = hamiltorch.Metric.HESSIAN,
                     debug: bool = False,
                     verbose: bool = False,
                     normalizing_const: int = 1,
                     inv_mass = None):
        
        "Use hamiltorch functions to run the Hamiltonian Monte Carlo algorithm to learn the parameters of the model and make predictions in the validation dataset"
        
        params_hmc = hamiltorch.samplers.sample_model(
            model = self.net, 
            x = X_tr, 
            y = Y_tr,
            model_loss = 'regression',
            num_samples=num_samples_hmc,
            num_steps_per_sample = L,
            step_size = step_size,
            burn = burn_sample,
            inv_mass = inv_mass,
            normalizing_const = normalizing_const,
            sampler = sampler,
            integrator = integrator,
            metric = metric,
            params_init = self.params_init, 
            tau_out = tau_out, 
            tau_list = tau_list)
        
        self._save_params_hmc(params_hmc)
        
        pred_list, log_prob_list = hamiltorch.predict_model(
            model=self.net,
            x = X_val, 
            y = Y_val, 
            model_loss = 'regression',
            samples=params_hmc[burn_pred:],
            tau_out=tau_out, 
            tau_list=tau_list)
        
        self._save_params_hmc(params_hmc)
        
        return params_hmc, pred_list, log_prob_list
