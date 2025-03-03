import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TransformerConv
from torch_geometric.typing import Adj


class GNNLayer(nn.Module):
    """Base GNN layer that can be configured to use different graph convolution types."""
    
    SUPPORTED_LAYERS = {
        "GCN": GCNConv,
        "GAT": GATConv,
        "GraphSAGE": SAGEConv,
        "GraphConv": GraphConv,
        "Transformer": TransformerConv
    }
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        layer_type: str = "GCN", 
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        residual: bool = False,
        **kwargs
    ):
        """
        Initialize a GNN layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            layer_type: Type of GNN layer ("GCN", "GAT", "GraphSAGE", etc.)
            activation: Activation function ("relu", "leaky_relu", "elu", "gelu", None)
            dropout: Dropout rate
            batch_norm: Whether to apply batch normalization
            residual: Whether to add a residual connection
            **kwargs: Additional arguments for the specific layer type
        """
        super().__init__()
        
        if layer_type not in self.SUPPORTED_LAYERS:
            raise ValueError(f"Unknown layer type: {layer_type}. Supported types: {list(self.SUPPORTED_LAYERS.keys())}")
            
        # Create the graph convolution layer
        self.conv = self.SUPPORTED_LAYERS[layer_type](in_channels, out_channels, **kwargs)
        
        # Setup activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        
        # Residual connection
        self.residual = residual
        if residual and in_channels != out_channels:
            self.res_linear = nn.Linear(in_channels, out_channels)
        else:
            self.res_linear = nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: Adj, **kwargs) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            **kwargs: Additional arguments for the specific layer type
            
        Returns:
            Updated node features
        """
        # Store input for residual connection
        identity = x
        
        # Apply graph convolution
        x = self.conv(x, edge_index, **kwargs)
        
        # Apply batch normalization
        x = self.batch_norm(x)
        
        # Apply activation function
        x = self.activation(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Add residual connection if requested
        if self.residual:
            x = x + self.res_linear(identity)
            
        return x