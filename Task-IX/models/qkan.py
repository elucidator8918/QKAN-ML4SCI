import torch
import torch.nn as nn
from models.layers.kan_linear import KANLinear
from models.layers.qkan_linear import QKANLinear

class QKAN(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN).
    A neural network composed of KANLinear layers.
    """
    
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        Initialize a Kolmogorov-Arnold Network.
        
        Args:
            layers_hidden (list): List of layer sizes, including input and output dimensions.
            grid_size (int): Size of the grid for spline computation.
            spline_order (int): Order of the spline.
            scale_noise (float): Scale of the noise for initialization.
            scale_base (float): Scale of the base weights.
            scale_spline (float): Scale of the spline weights.
            base_activation (nn.Module): Activation function for the base transformation.
            grid_eps (float): Epsilon for grid adaptation.
            grid_range (list): Range of the grid.
        """
        super(QKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create layers
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        self.layers.append(QKANLinear(
            out_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        ))

    def forward(self, x, update_grid=False):
        """
        Forward pass of the KAN.
        
        Args:
            x (torch.Tensor): Input tensor.
            update_grid (bool): Whether to update the grid.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss for the entire network.
        
        Args:
            regularize_activation (float): Regularization factor for activation.
            regularize_entropy (float): Regularization factor for entropy.
            
        Returns:
            torch.Tensor: Regularization loss.
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )