import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base_linear import BaseLinear

class KANLinear(BaseLinear):
    """
    Kolmogorov-Arnold Network Linear Layer as described in the original paper.
    This implements a neural network layer with spline-based activation functions.
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__(
            in_features,
            out_features,
            grid_size,
            spline_order,
            scale_noise,
            scale_base,
            scale_spline,
            enable_standalone_scale_spline,
            base_activation,
            grid_eps,
            grid_range,
        )
        
        # Create base weight parameter for linear transformation
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    
    def _initialize_parameters(self):
        """Initialize the layer parameters."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )
    
    def forward(self, x):
        """
        Forward pass of the KANLinear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Compute base and spline outputs
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # Combine outputs and reshape to original dimensions
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        
        return output