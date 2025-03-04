import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.typing import Adj
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score
from .gnn import GNN


class GraphClassifier(nn.Module):
    """Graph-level classification model that uses GNN for node embeddings followed by pooling."""
    
    POOLING_METHODS = {
        "mean": global_mean_pool,
        "max": global_max_pool,
        "sum": global_add_pool,
    }
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        gnn_params: dict = None,
        pooling: str = "mean",
        dropout: float = 0.5,
        final_layers: list = None
    ):
        """
        Initialize the graph classifier.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes
            gnn_params: Parameters for the GNN model
            pooling: Pooling method for graph-level readout
            dropout: Dropout rate for final classification layer
            final_layers: List of hidden layer sizes for MLP after pooling
        """
        super().__init__()
        
        # Default GNN parameters
        if gnn_params is None:
            gnn_params = {
                "num_layers": 3,
                "layer_type": "GCN",
                "dropout": 0.1,
                "batch_norm": True,
                "residual": True
            }
        
        # GNN for computing node embeddings
        self.gnn = GNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Output embeddings, not predictions
            **gnn_params
        )
        
        # Set up pooling method
        if pooling not in self.POOLING_METHODS:
            raise ValueError(f"Unknown pooling method: {pooling}")
        self.pooling_fn = self.POOLING_METHODS[pooling]
        
        # Build classification head (MLP)
        classifier_layers = []
        
        # Add intermediate layers if specified
        input_dim = hidden_channels
        if final_layers:
            for layer_size in final_layers:
                classifier_layers.extend([
                    nn.Linear(input_dim, layer_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                input_dim = layer_size
        
        # Add final classification layer
        classifier_layers.append(nn.Linear(input_dim, out_channels))
        
        # Create the classifier
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor, edge_index: Adj, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph classifier.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for nodes
            
        Returns:
            Classification logits
        """
        # Get node embeddings
        node_embeddings = self.gnn(x, edge_index)
        
        # Pool node embeddings to graph-level representation
        graph_embedding = self.pooling_fn(node_embeddings, batch)
        
        # Apply classification head
        return self.classifier(graph_embedding)


class JetClassifier(pl.LightningModule):
    """PyTorch Lightning module for jet classification with GNNs."""
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        gnn_params: dict = None,
        pooling: str = "mean",
        final_layers: list = None,
        lr_scheduler: str = "onecycle",
        lr_scheduler_params: dict = None,
        class_weights: list = None,
    ):
        """
        Initialize the jet classifier.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes (1 for binary)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            gnn_params: Parameters for the GNN component
            pooling: Pooling method for graph-level readout
            final_layers: List of hidden layer sizes for MLP after pooling
            lr_scheduler: Learning rate scheduler type
            lr_scheduler_params: Parameters for the learning rate scheduler
            class_weights: Weights for imbalanced classes
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the model
        self.model = GraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gnn_params=gnn_params,
            pooling=pooling,
            final_layers=final_layers
        )
        
        # Set up class weights for imbalanced data
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None
        
        # Loss function
        if out_channels == 1:  # Binary classification
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[1] if self.class_weights is not None else None)
        else:  # Multi-class classification
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Metrics
        self.train_metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'precision': Precision(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'recall': Recall(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'f1': F1Score(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'auc': AUROC(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels)
        })
        
        self.val_metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'precision': Precision(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'recall': Recall(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'f1': F1Score(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'auc': AUROC(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels)
        })
        
        self.test_metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'precision': Precision(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'recall': Recall(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'f1': F1Score(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels),
            'auc': AUROC(task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels)
        })
        
        # Save scheduler parameters
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}

    def forward(self, data):
        """
        Forward pass through the model.
        
        Args:
            data: PyG data object containing x, edge_index, and batch
            
        Returns:
            Model predictions
        """
        x = self.model(data.x, data.edge_index, data.batch)
        return x.squeeze(dim=-1) if x.shape[-1] == 1 else x

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                **self.lr_scheduler_params
            )
        elif self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                **self.lr_scheduler_params
            )
        elif self.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                **self.lr_scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_auc"
            }
        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def _log_metrics(self, prefix, metrics_dict, batch_y, logits, loss=None):
        """Helper method to log metrics for a given phase."""
        if loss is not None:
            self.log(f"{prefix}_loss", loss, prog_bar=(prefix == "val"), on_epoch=True, batch_size=batch_y.size(0))

        # Compute predictions from logits based on task type
        if self.hparams.out_channels == 1:  # Binary classification
            preds = torch.sigmoid(logits)
        else:  # Multi-class classification
            preds = F.softmax(logits, dim=-1)
        
        # Update and log all metrics
        for name, metric in metrics_dict.items():
            metric(preds, batch_y)
            self.log(
                f"{prefix}_{name}",
                metric,
                prog_bar=(prefix == "val" and name == "auc"),
                on_epoch=True,
                batch_size=batch_y.size(0)  # Specify correct batch size
            )

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits = self(batch)
        
        # Convert labels to appropriate format
        if self.hparams.out_channels == 1:
            target = batch.y.float()
        else:
            target = batch.y.long()
        
        # Calculate loss
        loss = self.loss_fn(logits, target)
        
        # Log metrics
        self._log_metrics("train", self.train_metrics, batch.y, logits, loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits = self(batch)
        
        # Convert labels to appropriate format
        if self.hparams.out_channels == 1:
            target = batch.y.float()
        else:
            target = batch.y.long()
        
        # Calculate loss
        loss = self.loss_fn(logits, target)
        
        # Log metrics
        self._log_metrics("val", self.val_metrics, batch.y, logits, loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits = self(batch)
        
        # Log metrics
        self._log_metrics("test", self.test_metrics, batch.y, logits)
