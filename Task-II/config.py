"""
Configuration module for jet classification project.
"""
from typing import Dict, Any, List, Optional
import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    # Data paths
    data_root: str = "jets"
    train_file: str = "QG_jets.npz"
    test_file: Optional[str] = None
    
    # Dataset parameters
    max_samples: Optional[int] = None  # None means use all available samples
    node_feature_dim: int = 4  # pT, y, phi, PID
    
    # DataLoader parameters
    batch_size: int = 256
    num_workers: int = 8
    train_val_test_split: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check that split proportions sum to 1
        if sum(self.train_val_test_split) != 1.0:
            raise ValueError("Train/val/test split must sum to 1.0")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # GNN parameters
    gnn_type: str = "GCN"  # "GCN", "GAT", etc.
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    
    # Layer-specific parameters
    gcn_params: Dict[str, Any] = field(default_factory=lambda: {
        "improved": True,
        "cached": False,
    })
    
    gat_params: Dict[str, Any] = field(default_factory=lambda: {
        "heads": 4,
        "concat": True,
        "negative_slope": 0.2,
    })
    
    # Graph pooling
    pooling: str = "mean"  # "mean", "max", or "sum"


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Optimizer parameters
    optimizer: str = "Adam"  # "Adam", "SGD", etc.
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    
    # Training loop parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Scheduler parameters
    use_lr_scheduler: bool = True
    scheduler_type: str = "OneCycleLR"  # "ReduceLROnPlateau", "StepLR"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 10000.0,
    })
    
    # Hardware
    use_gpu: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging and visualization."""
    
    # Logging directory
    log_dir: str = "logs"
    log_every_n_steps: int = 50
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    
    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: ["loss", "auc"])
    
    # Visualization
    create_visualizations: bool = True
    visualization_dir: str = "visualizations"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.create_visualizations:
            os.makedirs(self.visualization_dir, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Master configuration for an experiment."""
    
    # Experiment name
    experiment_name: str = "jet_classification"
    
    # Version (can be auto-generated with timestamp)
    version: Optional[str] = None
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Component configs
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "version": self.version,
            "seed": self.seed,
            "data": self.data_config.__dict__,
            "model": self.model_config.__dict__,
            "training": self.training_config.__dict__,
            "logging": self.logging_config.__dict__,
        }


# Default configurations for different experiment types
default_configs = {
    "gcn_baseline": ExperimentConfig(
        experiment_name="gcn_baseline",
        version="v1",
        model_config=ModelConfig(
            gnn_type="GCN", 
            hidden_channels=64, 
            num_layers=3, 
            dropout=0.2
        ),
        training_config=TrainingConfig(
            learning_rate=0.01,
            max_epochs=100
        )
    ),
    
    "gat_baseline": ExperimentConfig(
        experiment_name="gat_baseline",
        version="v1",
        model_config=ModelConfig(
            gnn_type="GAT", 
            hidden_channels=64, 
            num_layers=3, 
            dropout=0.2,
            gat_params={"heads": 4, "concat": True}
        ),
        training_config=TrainingConfig(
            learning_rate=0.005,
            max_epochs=100
        )
    ),
    
    "production": ExperimentConfig(
        experiment_name="production",
        version="v1",
        data_config=DataConfig(
            batch_size=512,
            num_workers=16
        ),
        model_config=ModelConfig(
            gnn_type="GCN", 
            hidden_channels=128, 
            num_layers=4, 
            dropout=0.3
        ),
        training_config=TrainingConfig(
            learning_rate=0.01,
            weight_decay=1e-4,
            max_epochs=200,
            early_stopping_patience=20
        )
    )
}