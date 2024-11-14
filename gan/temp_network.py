"""Temporal Network."""

from torch import Tensor

import torch
from torch import nn
from .utils import Reshape


class TemporalNetwork(nn.Module):
    """Temporal network.

    Parameters
    ----------
    z_dimension: int, (default=32)
        Noise space dimension.
    hid_channels: int, (default=1024)
        Number of hidden channels.

    """

    def __init__(
        self,
        z_dimension: int = 32, # ye spacesize  define karta hai 
        hid_channels: int = 1024, 
        n_bars: int = 2,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.n_bars = n_bars        # ye number of bars hai (holds a specific number of beats, creating a sense of structure and rhythm)
        self.net = nn.Sequential(   #sequential container  to stack different layers of the network.
            # input shape: (batch_size, z_dimension)
            Reshape(shape=[z_dimension, 1, 1]), #reshape the input tensor from a 2D format to a 4D format
            # output shape: (batch_size, z_dimension, 1, 1)
            nn.ConvTranspose2d( #increases the spatial dimensions
                z_dimension, #input channel size.
                hid_channels, #output channel size.
                kernel_size=(2, 1),# defines a 2x1 filter. ( kernel with dimensions 2 rows by 1)
                stride=(1, 1),
                padding=0, # no padding
            ),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),   # pura reLu function pe chal raha hai 
            # output shape: (batch_size, hid_channels, 2, 1)
            nn.ConvTranspose2d(
                hid_channels,
                z_dimension,
                kernel_size=(self.n_bars - 1, 1),     # se
                stride=(1, 1),
                padding=0,    
            ),
            nn.BatchNorm2d(z_dimension),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, z_dimension, 1, 1) 
            Reshape(shape=[z_dimension, self.n_bars]), # formats the output to a 2D shape [batch_size, z_dimension, n_bars], creating the structured temporal output.
        )

    def forward(self, x: Tensor) -> Tensor:  #  forward pass of the network:
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
        fx = self.net(x)
        return fx
