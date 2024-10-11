import torch
import pyro.infer.mcmc as mcmc
import logging
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Callable, Optional, Dict, Any


@beartype
class HMCTrainer:
    def __init__(
        self,
        model: Callable,
        step_size: float = 1.0,
        trajectory_length: Optional[float] = None,
        num_steps: Optional[int] = None,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        full_mass: bool = False,
        target_accept_prob: float = 0.8,
        max_plate_nesting: Optional[int] = None,
        warmup_steps: PositiveInteger = 100,
        num_samples: PositiveInteger = 500,
    ):
        """Prepare to train a model using Hamiltonian Monte Carlo (HMC)

        :param model: Pyro model to be trained using HMC
        :param step_size: Step size for the HMC sampler
        :param trajectory_length: Length of a MCMC trajectory (optional)
        :param num_steps: Number of discrete steps for Hamiltonian dynamics (optional)
        :param adapt_step_size: Flag to enable/disable step size adaptation during warm-up
        :param adapt_mass_matrix: Flag to enable/disable mass matrix adaptation during warm-up
        :param full_mass: Flag to specify if mass matrix is dense or diagonal
        :param target_accept_prob: Target acceptance probability for HMC
        :param max_plate_nesting: Maximum number of nested pyro.plate() contexts (optional)
        :param warmup_steps: Number of warm-up steps before sampling
        :param num_samples: Number of samples to draw after warm-up
        """
        self.model = model
        self.step_size = step_size
        self.trajectory_length = trajectory_length
        self.num_steps = num_steps
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.full_mass = full_mass
        self.target_accept_prob = target_accept_prob
        self.max_plate_nesting = max_plate_nesting
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples

        self.history: Dict[str, Optional[torch.Tensor]] = {
            "samples": None,
            "acceptance_rate": None,
            "step_size": None,
            "potential_energy": None,
        }

        self.logger = logging.getLogger(__name__)

    def run(
        self,
        init_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Run HMC to sample from the posterior distribution of the model

        :param init_params: Initial parameters for the model (optional)
        """
        hmc_kernel = mcmc.HMC(
            model=self.model,
            step_size=self.step_size,
            trajectory_length=self.trajectory_length,
            num_steps=self.num_steps,
            adapt_step_size=self.adapt_step_size,
            adapt_mass_matrix=self.adapt_mass_matrix,
            full_mass=self.full_mass,
            target_accept_prob=self.target_accept_prob,
            max_plate_nesting=self.max_plate_nesting,
        )

        mcmc_sampler = mcmc.MCMC(
            hmc_kernel,
            num_samples=self.num_samples,
            warmup_steps=self.warmup_steps,
            initial_params=init_params,
        )

        self.logger.info("Starting HMC sampling")
        mcmc_sampler.run()

        self.history["samples"] = mcmc_sampler.get_samples()
        self.history["acceptance_rate"] = torch.tensor(
            mcmc_sampler.diagnostics()["acceptance_rate"]
        )
        self.history["step_size"] = torch.tensor(
            mcmc_sampler.diagnostics()["step_size"]
        )
        self.history["potential_energy"] = torch.tensor(
            mcmc_sampler.diagnostics()["potential_energy"]
        )

        self.logger.info("HMC sampling completed")
