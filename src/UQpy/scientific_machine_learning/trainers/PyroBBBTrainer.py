import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
import pyro
from pyro.nn import PyroModule, PyroSample, PyroParam
import logging
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union, Type
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer import Predictive
from pyro.infer.util import (
    check_fully_reparametrized,
    is_validation_enabled,
    torch_item,
)
from torch.distributions import kl_divergence
from pyro.distributions.util import scale_and_mask
from typing import Callable, Union
from torch.distributions import constraints  # 파일 상단 import 구역에 추가
import pyro.distributions as dist


@beartype
class PyroBBBTrainer:
    
    def __init__(
        self,
        model: PyroModule,
        optimizer: pyro.optim.PyroOptim,
        scheduler: Union[pyro.optim.lr_scheduler.PyroLRScheduler, list] = None,
        loss_function: Callable = Trace_ELBO,
        guide: Union[Callable, pyro.infer.autoguide.AutoGuide] = AutoDiagonalNormal,):
        """Prepare to train a Bayesian neural network using Bayes by back propagation with Pyro

        :param model: Bayesian Neural Network model to be trained
        :param optimizer: Optimization algorithm used to update ``model`` parameters
        :param scheduler: Scheduler used to adjust the learning rate of the ``optimizer``.
         Schedulers may be chained together by creating a list of schedulers
        :param loss_function (DEPRECATED): Function used to compute negative log likelihood of the data during training
        :param divergence (DEPRECATED): Divergence measured between prior and posterior distribution of Bayesian layers
         Default: ``sml.GaussianKullbackLeiblerLoss()``
        """
        self.model = model
        pyro.clear_param_store()
        self.guide = guide
        self.optimizer = optimizer
        self.scheduler = (
            [scheduler]
            if isinstance(scheduler, pyro.optim.lr_scheduler.PyroLRScheduler)
            else scheduler
        )
        self.loss_function = loss_function

        self.history: dict = {
            "train_loss": torch.tensor(torch.nan),
            "train_divergence": torch.tensor(torch.nan),
            "train_nll": torch.tensor(torch.nan),
            "test_nll": torch.tensor(torch.nan),
        }
        """Record of the loss during training and validation. 
        Note if training ends early there may be ``NaN`` values, as the histories are initialized with ``NaN``.
        
        - ``history["train_loss"]`` contains the training history as a ``torch.Tensor``.
        - ``history["train_divergence"]`` contains the training divergence as a ``torch.Tensor``.
        - ``history["train_nll"]`` contains the training negative log likelihood loss as a ``torch.Tensor``.
        - ``history["test_nll"]`` contains the testing negative log likelihood loss as a ``torch.Tensor``.
         """
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        train_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
        epochs: PositiveInteger = 100,
        num_samples: PositiveInteger = 1,
        tolerance: float = 0.0,
    ):
        
        """Run the ''optimizer'' algorithm to learn the parameters of the ''model'' that fit ''train_data''

        :param train_data: Data used to compute ``model`` loss
        :param test_data: Data used to validate the performance of the model
        :param epochs: Maximum number of epochs to run the ``optimizer`` for
        :param num_samples: Number of Monte Carlo samples to approximate the loss (i.e., num_particles in Pyro)
        :param tolerance: Optimization terminates early if *average* training loss is below tolerance.
         Default: 0.0
        :param beta (DEPRECATED): Weighting for the divergence term in ELBO loss. Default: 1.0

        :raises RuntimeError: If neither ``train_data`` nor ``test_data`` is provided, a RuntimeError occurs.
        """
        if train_data and not test_data:
            log_note = f"training {self.model.__class__.__name__}"
        elif not train_data and test_data:
            log_note = f"testing {self.model.__class__.__name__}"
        elif train_data and test_data:
            log_note = f"training and testing {self.model.__class__.__name__}"
        else:
            raise RuntimeError(
                "UQpy: At least one of `train_data` or `test_data` must be provided."
            )
        

        if train_data:
            self.history["train_loss"] = torch.full(
                [epochs], torch.nan, requires_grad=False
            )
            # self.history["train_divergence"] = torch.full(
            #     [epochs], torch.nan, requires_grad=False
            # )
            # self.history["train_nll"] = torch.full(
            #     [epochs], torch.nan, requires_grad=False
            # )
        if test_data:
            self.history["test_nll"] = torch.full(
                [epochs], torch.nan, requires_grad=False
            )

        # svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO(num_particles=num_samples))
        svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss_function(num_particles=num_samples))
        self.logger.info("UQpy: Scientific Machine Learning: Beginning " + log_note)
        i = 0
        average_train_loss = torch.inf
        while i < epochs and average_train_loss > tolerance:
            if train_data:
                total_train_loss = 0
                # total_nll_loss = 0
                # total_divergence_loss = 0
                for batch_number, (*x, y) in enumerate(train_data):
                    train_loss = svi.step(*x, y) # - y.numel() * torch.log(torch.tensor(self.model.sigma * (2 * torch.pi)**0.5))
                    total_train_loss += train_loss
                if self.scheduler:
                    for s in self.scheduler:
                        if isinstance(s, pyro.optim.lr_scheduler.ReduceLROnPlateau):
                            s.step(train_loss)
                        else:
                            s.step()
                average_train_loss = total_train_loss / len(train_data)
                # average_train_nll = total_nll_loss / len(train_data)
                # average_train_divergence = total_divergence_loss / len(train_data)
                self.history["train_loss"][i] = average_train_loss
                # self.history["train_nll"][i] = average_train_nll
                # self.history["train_divergence"][i] = average_train_divergence
                self.model.eval()
            log_message = (
                f"UQpy: Scientific Machine Learning: "
                f"Epoch {i + 1:,} / {epochs:,} "
                f"Train Loss {average_train_loss:.6e} "
                # f"Train NLL {average_train_nll:.6e} "
                # f"Train Divergence {average_train_divergence:.6e} "
            )
            if test_data:
                total_test_nll = 0
                with torch.no_grad():
                    predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("linear.weight", "obs", "_RETURN", "sigma"))
                    for batch_number, (*x, y) in enumerate(test_data):
                        preds = predictive(*x)
                        test_prediction = torch.mean(preds["_RETURN"], dim=0)
                        test_nll = nn.MSELoss()(test_prediction, y)
                        total_test_nll += test_nll.item()
                average_test_nll = total_test_nll / len(test_data)
                self.history["test_nll"][i] = average_test_nll
                log_message += f" Test NLL {average_test_nll:.6e}"
            self.logger.info(log_message)

            i += 1
        
        self.logger.info(f"UQpy: Scientific Machine Learning: Completed " + log_note)

# We need to manually define a guide function for analytical computing of KL divergence between prior and variational distribution
# When we do not need analytical KL divergence, we can use Pyro's built-in AutoGuide functions
def mean_field_guide_factory(model, posterior_mu_initial=(0.0, 0.1), posterior_rho_initial=(-3.0, 0.1)):
    linear_layers = [
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("layer5", model.layer5),
    ]

    def draw_normal(shape, init):
        mean, std = init
        return torch.empty(shape).normal_(mean=mean, std=std)

    def guide(x, y=None):
        for name, layer in linear_layers:
            weight_shape = (layer.out_features, layer.in_features)
            bias_shape = (layer.out_features,)

            weight_loc = pyro.param(
                f"{name}_weight_loc",
                draw_normal(weight_shape, posterior_mu_initial),
            )
            weight_scale = pyro.param(
                f"{name}_weight_scale",
                torch.log1p(torch.exp(draw_normal(weight_shape, posterior_rho_initial))),
                constraint=constraints.positive,
            )
            pyro.sample(
                f"{name}.weight",
                dist.Normal(weight_loc, weight_scale).to_event(2),
            )

            bias_loc = pyro.param(
                f"{name}_bias_loc",
                draw_normal(bias_shape, posterior_mu_initial),
            )
            bias_scale = pyro.param(
                f"{name}_bias_scale",
                torch.log1p(torch.exp(draw_normal(bias_shape, posterior_rho_initial))),
                constraint=constraints.positive,
            )
            pyro.sample(
                f"{name}.bias",
                dist.Normal(bias_loc, bias_scale).to_event(1),
            )

    return guide


# We create a custom TraceMeanField_ELBO class to log a warning message when KL divergence is computed via sampling
# instead of analytically. This is useful for debugging purposes.
class TraceMeanField_ELBO_experimental(TraceMeanField_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
    
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        blank_kl = 0.0
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                else:
                    guide_site = guide_trace.nodes[name]
                    if is_validation_enabled():
                        check_fully_reparametrized(guide_site)

                    # use kl divergence if available, else fall back on sampling
                    try:
                        kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                        kl_qp = scale_and_mask(
                            kl_qp, scale=guide_site["scale"], mask=guide_site["mask"]
                        )
                        if torch.is_tensor(kl_qp):
                            assert (
                                torch._C._get_tracing_state()
                                or kl_qp.shape == guide_site["fn"].batch_shape
                            )
                            kl_qp_sum = kl_qp.sum()
                        else:
                            kl_qp_sum = (
                                kl_qp * torch.Size(guide_site["fn"].batch_shape).numel()
                            )
                        blank_kl += kl_qp_sum
                        elbo_particle = elbo_particle - kl_qp_sum
                    except NotImplementedError:
                        self.logger.info("UQpy: Falling back to sampling for KL term.")
                        entropy_term = guide_site["score_parts"].entropy_term
                        elbo_particle = (
                            elbo_particle
                            + model_site["log_prob_sum"]
                            - entropy_term.sum()
                        )

        # handle auxiliary sites in the guide
        blank_entropy = 0.0
        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample" and name not in model_trace.nodes:
                assert guide_site["infer"].get("is_auxiliary")
                if is_validation_enabled():
                    check_fully_reparametrized(guide_site)
                entropy_term = guide_site["score_parts"].entropy_term
                elbo_particle = elbo_particle - entropy_term.sum()
                blank_entropy += entropy_term.sum()

        if blank_entropy + blank_kl < 0.0:
            self.logger.warning(
                f"UQpy: ELBO entropy + KL term is negative ({blank_entropy + blank_kl}). This may indicate an incorrectly specified model or guide."
            )
        loss = -(
            elbo_particle.detach()
            if torch._C._get_tracing_state()
            else torch_item(elbo_particle)
        )
        surrogate_loss = -elbo_particle
        return loss, surrogate_loss    


