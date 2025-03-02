import torch
import math

def create_grid(in_features, grid_size, spline_order, grid_range=[-1, 1]):
    """
    Create a grid for spline computation.
    
    Args:
        in_features (int): Number of input features.
        grid_size (int): Size of the grid.
        spline_order (int): Order of the spline.
        grid_range (list): Range of the grid.
        
    Returns:
        torch.Tensor: Grid tensor.
    """
    h = (grid_range[1] - grid_range[0]) / grid_size
    return (
        (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        .expand(in_features, -1)
        .contiguous()
    )

def compute_b_splines(x, grid, in_features, grid_size, spline_order):
    """
    Compute the B-spline bases for the given input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        grid (torch.Tensor): Grid tensor.
        in_features (int): Number of input features.
        grid_size (int): Size of the grid.
        spline_order (int): Order of the spline.

    Returns:
        torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
    """
    assert x.dim() == 2 and x.size(1) == in_features

    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )

    assert bases.size() == (
        x.size(0),
        in_features,
        grid_size + spline_order,
    )
    return bases.contiguous()

def curve_to_coeff(x, y, in_features, out_features, grid_size, spline_order, b_splines_func):
    """
    Compute the coefficients of the curve that interpolates the given points.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        grid_size (int): Size of the grid.
        spline_order (int): Order of the spline.
        b_splines_func (callable): Function to compute B-splines.

    Returns:
        torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    """
    assert x.dim() == 2 and x.size(1) == in_features
    assert y.size() == (x.size(0), in_features, out_features)

    A = b_splines_func(x).transpose(
        0, 1
    )  # (in_features, batch_size, grid_size + spline_order)
    B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
    solution = torch.linalg.lstsq(
        A, B
    ).solution  # (in_features, grid_size + spline_order, out_features)
    result = solution.permute(
        2, 0, 1
    )  # (out_features, in_features, grid_size + spline_order)

    assert result.size() == (
        out_features,
        in_features,
        grid_size + spline_order,
    )
    return result.contiguous()

def update_grid_and_weights(
    x, layer, margin=0.01
):
    """
    Update the grid and spline weights based on the input data distribution.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        layer: The layer whose grid and weights need to be updated.
        margin (float): Margin for the grid.
    """
    batch = x.size(0)
    in_features = layer.in_features
    grid_size = layer.grid_size
    spline_order = layer.spline_order

    splines = layer.b_splines(x)  # (batch, in, coeff)
    splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
    orig_coeff = layer.scaled_spline_weight  # (out, in, coeff)
    orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
    unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
    unreduced_spline_output = unreduced_spline_output.permute(
        1, 0, 2
    )  # (batch, in, out)

    # sort each channel individually to collect data distribution
    x_sorted = torch.sort(x, dim=0)[0]
    grid_adaptive = x_sorted[
        torch.linspace(
            0, batch - 1, grid_size + 1, dtype=torch.int64, device=x.device
        )
    ]

    uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size
    grid_uniform = (
        torch.arange(
            grid_size + 1, dtype=torch.float32, device=x.device
        ).unsqueeze(1)
        * uniform_step
        + x_sorted[0]
        - margin
    )

    grid = layer.grid_eps * grid_uniform + (1 - layer.grid_eps) * grid_adaptive
    grid = torch.concatenate(
        [
            grid[:1]
            - uniform_step
            * torch.arange(spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:]
            + uniform_step
            * torch.arange(1, spline_order + 1, device=x.device).unsqueeze(1),
        ],
        dim=0,
    )

    layer.grid.copy_(grid.T)
    layer.spline_weight.data.copy_(layer.curve2coeff(x, unreduced_spline_output))

def compute_regularization_loss(spline_weight, regularize_activation=1.0, regularize_entropy=1.0):
    """
    Compute the regularization loss.
    
    Args:
        spline_weight (torch.Tensor): Spline weights.
        regularize_activation (float): Regularization factor for activation.
        regularize_entropy (float): Regularization factor for entropy.
        
    Returns:
        torch.Tensor: Regularization loss.
    """
    l1_fake = spline_weight.abs().mean(-1)
    regularization_loss_activation = l1_fake.sum()
    p = l1_fake / regularization_loss_activation
    regularization_loss_entropy = -torch.sum(p * p.log())
    return (
        regularize_activation * regularization_loss_activation
        + regularize_entropy * regularization_loss_entropy
    )