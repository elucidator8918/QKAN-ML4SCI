import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils.spline_utils import (
    create_grid,
    compute_b_splines,
    curve_to_coeff,
    update_grid_and_weights,
    compute_regularization_loss
)

class BaseLinear(nn.Module):
    """Base class for KAN linear layers with spline-based activation."""
    
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
        super(BaseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Create and register the grid
        self.register_buffer("grid", create_grid(in_features, grid_size, spline_order, grid_range))
        
        # Create spline weights
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Optional standalone scaler for spline weights
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        # Store configuration parameters
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize the layer parameters. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_parameters")
    
    def b_splines(self, x):
        """
        Compute the B-spline bases for the given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            
        Returns:
            torch.Tensor: B-spline bases tensor.
        """
        return compute_b_splines(
            x, self.grid, self.in_features, self.grid_size, self.spline_order
        )
    
    def curve2coeff(self, x, y):
        """
        Compute the coefficients of the curve that interpolates the given points.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
            
        Returns:
            torch.Tensor: Coefficients tensor.
        """
        return curve_to_coeff(
            x, y, self.in_features, self.out_features, 
            self.grid_size, self.spline_order, self.b_splines
        )
    
    @property
    def scaled_spline_weight(self):
        """Get the scaled spline weights."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
    
    def forward(self, x):
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward")
    
    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        """
        Update the grid and spline weights based on the input data distribution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            margin (float): Margin for the grid.
        """
        update_grid_and_weights(x, self, margin)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        
        Args:
            regularize_activation (float): Regularization factor for activation.
            regularize_entropy (float): Regularization factor for entropy.
            
        Returns:
            torch.Tensor: Regularization loss.
        """
        return compute_regularization_loss(
            self.spline_weight, regularize_activation, regularize_entropy
        )