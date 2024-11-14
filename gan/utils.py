"""Utils."""

from typing import List
from torch import Tensor

import torch
from torch import nn


def initialize_weights(layer: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Initialize module with normal distribution.

    Parameters
    ----------
    layer: nn.Module
        Layer.
    mean: float, (default=0.0)
        Mean value.
    std: float, (default=0.02)
        Standard deviation value.

    """
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)


class Reshape(nn.Module):
    """Reshape layer.

    Parameters
    ----------
    shape: List[int]
        Dimensions after number of batches.

    """

    def __init__(self, shape: List[int]) -> None:
        """Initialize."""
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward.

        Parameters
        ----------
        x: Tensor
            Input batch.

        Returns
        -------
        Tensor:
            Preprocessed input batch.

        """
        return x.view(x.size(0), *self.shape)

class WassersteinLoss(nn.Module):
    """Wasserstein Loss."""
    def forward(self, real_output, fake_output):
        """Compute Wasserstein loss."""
        return torch.mean(fake_output) - torch.mean(real_output)

class GradientPenalty(nn.Module):
    """Gradient Penalty."""
    def __init__(self, lambda_gp: float = 10.0):
        """Initialize."""
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, real_samples, fake_samples, real_output):
        """Compute gradient penalty."""
        gradients = torch.autograd.grad(
            outputs=real_output,
            inputs=real_samples,
            grad_outputs=torch.ones_like(real_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = self.lambda_gp * torch.mean((gradients_norm - 1) ** 2)
        return gradient_penalty