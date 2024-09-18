import os
import torch
import torch.nn as nn
import hamiltorch
from beartype import beartype


@beartype
class HMCTrainer:
    def __init__(
        self, model: nn.Module, filename: str, loss_function: nn.Module = nn.MSELoss()
    ):
        """Prepare to train a model using Hamiltonian Monte Carlo (HMC)

        :param model:
        :param filename:
        :param loss_function: Function used to compute negative log likelihood of the data during training.
        """
        self.model = model
        self.filename = filename
        self.loss_function = loss_function
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"UQpy: Could not find file {self.filename!r}")
        self.initial_parameters = torch.load(filename)
        """Initial HMC parameters"""

    def run(
        self,
        X_tr: torch.Tensor = None,
        Y_tr: torch.Tensor = None,
        num_samples: int = 1_000,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
        burn: int = 0,
        sampler: hamiltorch.Sampler = hamiltorch.Sampler.HMC,
        tau_out: float = 1.0,
        tau_list: torch.Tensor = None,
        tau: float = 1.0,
        store_on_gpu: bool = False,
        save_path: str = "",
    ):
        """Sample weights from a NN model to perform inference.

        This function builds a log_prob_func from the torch.nn.Module and passes it to ``hamiltorch.sample``.

        :param X_tr: Input training data to define the log probability. Should be a shape that can be passed into the model.
         First dimension is :math:`N`, the number of data points.
        :param Y_tr: Output training data to define the log probability.
         Should be a shape that suits the likelihood (or - loss) of the model.
          First dimension is :math:`N`, the number of data points.
        :param num_samples: Sets the number of samples corresponding to the number of momentum resampling steps/the number of trajectories to sample.
        :param num_steps_per_sample: The number of steps to take per trajectory (often referred to as L).
        :param step_size: Size of each step to take when doing the numerical integration.
        :param burn: Number of samples to burn before collecting samples.
         Set to -1 for no burning of samples. This must be less than `num_samples` as `num_samples` subsumes `burn`.
        :param sampler: Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
        :param tau_out: Only relevant for model_loss = 'regression' (otherwise leave as 1.0).
         This corresponds the likelihood output precision.
        :param tau_list: A tensor containing the corresponding prior precision for each set of per layer parameters.
         This is assuming a Gaussian prior.
        :param tau:
        :param store_on_gpu: Option that determines whether to keep samples in GPU memory.
         It runs fast when set to TRUE but may run out of memory unless set to FALSE.
        :param save_path:
        """
        self.model.train()
        hmc_parameters = hamiltorch.sample_model(
            self.net,
            X_tr,
            Y_tr,
            params_init=self.params_init,
            model_loss=self.loss_function,
            num_samples=num_samples,
            num_steps_per_sample=num_steps_per_sample,
            step_size=step_size,
            burn=burn,
            sampler=sampler,
            tau_out=tau_out,
            tau_list=tau_list,
            tau=tau,
            store_on_GPU=store_on_gpu,
        )
        if save_path:
            torch.save(hmc_parameters, save_path)
        return hmc_parameters
