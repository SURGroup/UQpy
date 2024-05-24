import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass.Layer import Layer


class BayesianLayer(Layer):
    def __init__(self, weight_shape, bias_shape, priors, sampling=True, **kwargs):
        """

        :param weight_shape: Shape of the weight_mu and weight_sigma matrices
        :param bias_shape: Shpe of the bias_mu and bias_sigma matrices
        :param priors: Parameters for prior and posterior distributions
        :param sampling:
        """
        super().__init__(**kwargs)
        self.sampling = sampling
        self.priors = priors
        if self.priors is None:
            priors = {
                "prior_mu": 0,
                "prior_sigma": 0.1,
                "posterior_mu_initial": (0, 0.1),
                "posterior_rho_initial": (-3, 0.1),
            }
        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_initial = priors["posterior_mu_initial"]
        self.posterior_rho_initial = priors["posterior_rho_initial"]

        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_sigma = nn.Parameter(torch.empty(weight_shape))

        self.bias = True if bias_shape else False
        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(bias_shape))
            self.bias_sigma = nn.Parameter(torch.empty(bias_shape))
        else:
            self.bias_mu = None
            self.bias_sigma = None
        self._sample_parameters()

    def sample(self, mode: bool = True):
        """Set sampling mode.

        :param mode: If ``True``, layer parameters are sampled from their distributions.
         If ``False``, layer parameters are set to their means and layer acts deterministically.
        """
        self.sampling = mode

    def _sample_parameters(self):
        """Randomly populate weights and biases with samples from Normal distributions"""
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_sigma.data.normal_(*self.posterior_rho_initial)
        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_sigma.data.normal_(*self.posterior_rho_initial)

    def get_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :return:
        """
        if (
            self.training or self.sampling
        ):  # Randomly sample weights and biases from normal distribution
            weight_epsilon = torch.empty(self.weight_mu.size()).normal_(0, 1)
            w_sigma = torch.log1p(torch.exp(self.weight_sigma))
            weights = self.weight_mu + (weight_epsilon * w_sigma)
            if self.bias:
                bias_epsilon = torch.empty(self.bias_mu.size()).normal_(0, 1)
                b_sigma = torch.log1p(torch.exp(self.bias_sigma))
                biases = self.bias_mu + (bias_epsilon * b_sigma)
            else:
                biases = None
        else:  # Use mean values for weights and biases
            weights = self.weight_mu
            biases = self.bias_mu if self.bias else None
        return weights, biases

    def forward(self, *args, **kwargs):
        pass

    def extra_repr(self):
        pass
