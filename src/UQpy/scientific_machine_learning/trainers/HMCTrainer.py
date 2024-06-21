import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from torch import nn
import torch
import os
from enum import Enum
import hamiltorch
from hamiltorch import sample_model, comp_tau_list
from utils.utils_process import construct_paths
from beartype import beartype


@beartype
class HMCTrainer:
    def __init__(self, net: nn.Module, device: torch.device, params_hmc_path: str = None):
        self.net = net.to(device)
        self.device = device
        self.params_hmc_path = params_hmc_path
        self.params_init = self._initialize_params()

    def _initialize_params(self):
        if self.params_hmc_path and os.path.exists(self.params_hmc_path):
            return torch.load(self.params_hmc_path, map_location=self.device)[0]
        else:
            return hamiltorch.util.flatten(self.net).to(self.device).clone()

    def _perform_hmc_sampling(
        self,
        params_init: torch.Tensor = None,
        X_tr: torch.Tensor = None,
        Y_tr: torch.Tensor = None,
        X_ts: torch.Tensor = None,
        Y_ts: torch.Tensor = None,
        loss_function: nn.Module = nn.MSELoss(),
        num_samples_hmc: int = 1000,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
        burn: int = 0,
        inv_mass: torch.Tensor = None,
        jitter: float = None,
        normalizing_const: float = 1.,
        softabs_const: float = None,
        explicit_binding_const: float = 100,
        fixed_point_threshold: float = 1e-5,
        fixed_point_max_iterations: int = 1000,
        jitter_max_tries: int = 10,
        sampler: hamiltorch.Sampler = hamiltorch.Sampler.HMC,
        integrator: hamiltorch.Integrator = hamiltorch.Integrator.IMPLICIT,
        metric: hamiltorch.Metric = hamiltorch.Metric.HESSIAN,
        debug: bool = False,
        tau_out: float = 1.,
        tau_list: torch.Tensor = None,
        tau: float = 1.0,
        store_on_GPU: bool = False,
        desired_accept_rate: float = 0.8,
        verbose: bool = False,
    ):
        """
        Sample weights from a NN model to perform inference. This function builds a log_prob_func from the torch.nn.Module and passes it to `hamiltorch.sample`.

        :param params_init: torch.Tensor
            Initialisation of the parameters. This is a vector corresponding to the starting point of the sampler: shape: (D,), where D is the number of parameters of the model. The device determines which piece of hardware to run on.
        :param X_tr: torch.Tensor
            Input training data to define the log probability. Should be a shape that can be passed into the model. First dimension is N, where N is the number of data points.
        :param Y_tr: torch.Tensor
            Output training data to define the log probability. Should be a shape that suits the likelihood (or - loss) of the model. First dimension is N, where N is the number of data points.
        :param X_ts: torch.Tensor
            Input test data to define the log probability. Should be a shape that can be passed into the model. First dimension is N, where N is the number of data points.
        :param Y_ts: torch.Tensor
            Output test data to define the log probability. Should be a shape that suits the likelihood (or - loss) of the model. First dimension is N, where N is the number of data points.
        :param loss_function: nn.Module
            Function used to compute negative log likelihood of the data during training.
        :param num_samples_hmc: int
            Sets the number of samples corresponding to the number of momentum resampling steps/the number of trajectories to sample.
        :param num_steps_per_sample: int
            The number of steps to take per trajectory (often referred to as L).
        :param step_size: float
            Size of each step to take when doing the numerical integration.
        :param burn: int
            Number of samples to burn before collecting samples. Set to -1 for no burning of samples. This must be less than `num_samples` as `num_samples` subsumes `burn`.
        :param inv_mass: torch.Tensor or list
            The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
        :param jitter: float
            Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted. `jitter` is a float corresponding to scale of random draws from a uniform distribution.
        :param normalizing_const: float
            This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
        :param softabs_const: float
            Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
        :param explicit_binding_const: float
            Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
        :param fixed_point_threshold: float
            Only relevant for Implicit RMHMC. Sets the convergence threshold for 'breaking out' of the while loop for the generalised leapfrog.
        :param fixed_point_max_iterations: int
            Only relevant for Implicit RMHMC. Limits the number of fixed point iterations in the generalised leapfrog.
        :param jitter_max_tries: float
            Only relevant for RMHMC. Number of attempts at resampling the jitter for the Fisher Information before raising a LogProbError.
        :param sampler: hamiltorch.Sampler
            Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
        :param integrator: hamiltorch.Integrator
            Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING, Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
        :param metric: hamiltorch.Metric
            Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.
        :param debug: bool
            Debug mode can take 3 options. Setting debug = False (default) allows the sampler to run as normal. Setting debug = True prints both the old and new Hamiltonians per iteration, and also prints the convergence values when using the generalised leapfrog (IMPLICIT RMHMC).
        :param tau_out: float
            Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
        :param tau_list: torch.Tensor
            A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.
        :param store_on_GPU: bool
            Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
        :param desired_accept_rate: float
            Only relevant for NUTS. Sets the ideal acceptance rate that the NUTS will converge to.
        :param verbose: bool
            If set to True then do not display loading bar.
        """
        
        return hamiltorch.sample_model(
            self.net, 
            X_tr, 
            Y_tr,
            params_init = self.params_init, 
            model_loss = loss_function, 
            num_samples = num_samples_hmc, 
            burn = burn,
            step_size = step_size, 
            num_steps_per_sample = num_steps_per_sample, 
            tau_out = tau_out,
            tau_list = tau_list, 
            normalizing_const = normalizing_const,
            store_on_GPU = store_on_GPU,
            sampler = sampler, 
            #X_batch = X_ts_tensor.to(self.device), 
            #Y_batch= Y_ts_tensor.to(self.device), 
            #data_path= data_path, 
            tau = tau)

    def _save_params_hmc(self, params_hmc, params_hmc_path: str = None):
        if params_hmc_path is None:
            params_hmc_path = self.params_hmc_path
        torch.save(params_hmc, params_hmc_path)
        
    def run_training(self):
        self.net.train()
        # Prepare initial parameters for HMC
        params_init = self._initialize_params()
        # HMC Sampling
        params_hmc = self._perform_hmc_sampling()
        # Save parameters after sampling
        self._save_params_hmc(params_hmc)

        return params_hmc
