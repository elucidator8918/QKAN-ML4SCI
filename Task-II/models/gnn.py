import torch
import torch.nn as nn
from torch_geometric.typing import Adj
from .layers import GNNLayer
from typing import List, Optional, Dict, Union


class GNN(nn.Module):
    """Graph Neural Network model for processing graph-structured data."""
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        num_layers: int = 2, 
        layer_type: str = "GCN", 
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True,
        residual: bool = False,
        layer_config: Optional[Dict] = None,
        jk_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize GNN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GNN layers
            layer_type: Type of GNN layer to use
            dropout: Dropout rate
            activation: Activation function
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            layer_config: Additional configuration for each layer
            jk_mode: Jumping Knowledge mode (None, "cat", "max", "lstm")
            **kwargs: Additional arguments for GNN layers
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        
        # Default layer configuration
        if layer_config is None:
            layer_config = {}
        
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        first_layer_config = {**kwargs, **layer_config.get(0, {})}
        self.layers.append(
            GNNLayer(
                in_channels, 
                hidden_channels, 
                layer_type=layer_type, 
                activation=activation, 
                dropout=dropout,
                batch_norm=batch_norm,
                residual=False,  # First layer can't have residual
                **first_layer_config
            )
        )
        
        # Hidden layers: hidden_dim -> hidden_dim
        for i in range(1, num_layers - 1):
            layer_specific_config = {**kwargs, **layer_config.get(i, {})}
            self.layers.append(
                GNNLayer(
                    hidden_channels, 
                    hidden_channels, 
                    layer_type=layer_type, 
                    activation=activation, 
                    dropout=dropout,
                    batch_norm=batch_norm,
                    residual=residual,
                    **layer_specific_config
                )
            )
            
        # Output layer: hidden_dim -> out_dim (no activation after last layer)
        last_layer_config = {**kwargs, **layer_config.get(num_layers - 1, {})}
        self.layers.append(
            GNNLayer(
                hidden_channels, 
                out_channels, 
                layer_type=layer_type, 
                activation=None, 
                dropout=0,  # No dropout in final layer
                batch_norm=False,  # No batch norm in final layer
                residual=False,  # No residual in final layer
                **last_layer_config
            )
        )
        
        # Jumping Knowledge setup if requested
        if jk_mode == "cat":
            self.jk_linear = nn.Linear(hidden_channels * (num_layers - 1) + out_channels, out_channels)
        elif jk_mode == "max":
            # Max pooling doesn't need parameters
            pass
        elif jk_mode == "lstm":
            self.jk_lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True, bidirectional=True)
            self.jk_linear = nn.Linear(2 * hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: Adj, **kwargs) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            **kwargs: Additional arguments for GNN layers
            
        Returns:
            Updated node features
        """
        # For Jumping Knowledge, store intermediate representations
        if self.jk_mode is not None:
            intermediate = []
        
        # Process through GNN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, **kwargs)
            
            # For Jumping Knowledge, store intermediate layer outputs
            if self.jk_mode is not None and i < self.num_layers - 1:
                intermediate.append(x)
                
        # Apply Jumping Knowledge if requested
        if self.jk_mode is not None:
            # Add final layer output
            intermediate.append(x)
            
            if self.jk_mode == "cat":
                # Concatenate all representations
                x = torch.cat(intermediate, dim=1)
                x = self.jk_linear(x)
            elif self.jk_mode == "max":
                # Max pooling across layers
                x = torch.stack(intermediate, dim=0).max(dim=0)[0]
            elif self.jk_mode == "lstm":
                # Use LSTM to combine layer representations
                lstm_input = torch.stack(intermediate, dim=1)  # [num_nodes, num_layers, hidden_dim]
                lstm_out, _ = self.jk_lstm(lstm_input)
                x = lstm_out[:, -1]  # Use final LSTM output
                x = self.jk_linear(x)
        
        return x