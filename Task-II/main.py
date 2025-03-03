import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import local modules
from config import ExperimentConfig, default_configs
from utils.dataset import JetDataset
from models.classifier import JetClassifier
from utils.training import create_data_loaders, setup_trainer, train_model, evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Jet classification using GNNs")
    
    # Experiment configuration
    parser.add_argument('--config', type=str, default='gcn_baseline', 
                        choices=list(default_configs.keys()),
                        help='Configuration to use from default configs')
    parser.add_argument('--custom_config', type=str, 
                        help='Path to custom configuration file (overrides --config)')
    
    # Data options
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use')
    
    # Model options
    parser.add_argument('--gnn_type', type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSAGE'],
                        help='GNN architecture type (overrides config)')
    parser.add_argument('--pooling', type=str, choices=['mean', 'max', 'sum', 'attention'],
                        help='Graph pooling method (overrides config)')
    
    # Training options
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    
    # Experiment name
    parser.add_argument('--experiment_name', type=str, help='Experiment name (overrides config)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test'],
                        help='Mode to run in')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load for testing')
    
    return parser.parse_args()

def load_config(args):
    """Load configuration from args."""
    if args.custom_config and os.path.exists(args.custom_config):
        # Load custom config from file
        import json
        with open(args.custom_config, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig()
        # TODO: Properly load config from dict
    else:
        # Use default config
        config = default_configs[args.config]
    
    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_root:
        config.data_config.data_root = args.data_root
    if args.max_samples is not None:
        config.data_config.max_samples = args.max_samples
    if args.gnn_type:
        config.model_config.gnn_type = args.gnn_type
    if args.pooling:
        config.model_config.pooling = args.pooling
    if args.batch_size:
        config.data_config.batch_size = args.batch_size
    if args.learning_rate:
        config.training_config.learning_rate = args.learning_rate
    if args.max_epochs:
        config.training_config.max_epochs = args.max_epochs
    if args.seed:
        config.seed = args.seed
    
    return config

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seeds
    set_seeds(config.seed)
    
    # Print configuration summary
    print(f"Running experiment: {config.experiment_name}")
    print(f"Model: {config.model_config.gnn_type}")
    print(f"Using {'GPU' if config.training_config.use_gpu and torch.cuda.is_available() else 'CPU'}")
    
    # Create dataset
    dataset = JetDataset(
        root=config.data_config.data_root,
        filename=config.data_config.train_file,
        stop=config.data_config.max_samples,
        edge_strategy="physics",  # Can be made configurable
        normalize_features=True
    )
    
    # Create test dataset if specified
    test_dataset = None
    if config.data_config.test_file:
        test_dataset = JetDataset(
            root=config.data_config.data_root,
            filename=config.data_config.test_file,
            stop=config.data_config.max_samples,
            test=True,
            edge_strategy="physics",
            normalize_features=True
        )
    
    # Print dataset info
    print(f"Dataset: {len(dataset)} samples")
    print(f"Node features: {dataset.num_node_features} dimensions")
    
    if args.mode in ['train', 'train_test']:
        # Prepare model parameters
        model_kwargs = {
            'in_channels': dataset.num_node_features,
            'hidden_channels': config.model_config.hidden_channels,
            'out_channels': 1,  # Binary classification
            'learning_rate': config.training_config.learning_rate,
            'weight_decay': config.training_config.weight_decay,
            'gnn_params': {
                'layer_type': config.model_config.gnn_type,
                'num_layers': config.model_config.num_layers,
                'dropout': config.model_config.dropout,
                'batch_norm': True,
                'residual': True,
                'layer_params': config.model_config.gcn_params if config.model_config.gnn_type == 'GCN' else config.model_config.gat_params
            },
            'pooling': config.model_config.pooling,
            'final_layers': [128, 64],  # MLP after pooling
            'lr_scheduler': config.training_config.scheduler_type.lower(),
            'lr_scheduler_params': config.training_config.scheduler_params
        }
        
        # Training parameters
        train_kwargs = {
            'max_epochs': config.training_config.max_epochs,
            'gpus': args.gpus if config.training_config.use_gpu and torch.cuda.is_available() else 0,
            'train_ratio': config.data_config.train_val_test_split[0],
            'val_ratio': config.data_config.train_val_test_split[1],
            'num_workers': config.data_config.num_workers,
            'early_stopping_patience': config.training_config.early_stopping_patience,
            'monitor_metric': 'val_auc',  # Use AUC for monitoring
            'monitor_mode': 'max',
            'log_dir': config.logging_config.log_dir
        }
        
        # Train model
        model, metrics = train_model(
            model_class=JetClassifier,
            model_kwargs=model_kwargs,
            dataset=dataset,
            experiment_name=config.experiment_name,
            batch_size=config.data_config.batch_size,
            train_kwargs=train_kwargs,
            save_config=True
        )
        
        # Print training results
        print("Training completed!")
        print(f"Best validation AUC: {metrics.get('val_auc', 'N/A')}")
        print(f"Test AUC: {metrics.get('test_auc', 'N/A')}")
        
    if args.mode == 'test' or (args.mode == 'train_test' and test_dataset):
        # Load model from checkpoint if in test mode
        if args.mode == 'test':
            if not args.checkpoint:
                raise ValueError("Checkpoint must be provided for test mode")
            
            model = JetClassifier.load_from_checkpoint(args.checkpoint)
            print(f"Loaded model from {args.checkpoint}")
        
        # Test on separate test dataset if available
        if test_dataset:
            print("Evaluating on separate test dataset...")
            test_metrics = evaluate_model(
                model=model,
                dataset=test_dataset,
                batch_size=config.data_config.batch_size,
                num_workers=config.data_config.num_workers
            )
            
            # Print test results
            print("Test Results:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()