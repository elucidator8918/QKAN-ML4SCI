import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pennylane as qml
from base_linear import BaseLinear

class QKANLinear(BaseLinear):
    """
    Quantum Kolmogorov-Arnold Network Linear Layer.
    This implements a neural network layer with quantum computing for the base transformation
    and spline-based activation functions.
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
        # Setup quantum components before calling parent constructor
        self.n_qubits = max(in_features, out_features)
        self.n_layers = min(in_features, out_features)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        
        # Create quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encode inputs into quantum states
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply parameterized quantum gates
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RZ(weights[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Measure outputs
            return [qml.expval(qml.PauliZ(i)) for i in range(out_features)]
        
        # Create quantum layer
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Call parent constructor
        super(QKANLinear, self).__init__(
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
    
    def _initialize_parameters(self):
        """Initialize the layer parameters."""
        # Initialize quantum circuit weights
        nn.init.kaiming_uniform_(
            self.q_layer.quantum_circuit.weight, 
            a=math.sqrt(5) * self.scale_base
        )
        
        # Initialize spline weights
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
        Forward pass of the QKANLinear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Apply quantum circuit to get base output
        base_output = self.q_layer(self.base_activation(x))
        
        # Compute spline output
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # Combine outputs and reshape to original dimensions
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        
        return output