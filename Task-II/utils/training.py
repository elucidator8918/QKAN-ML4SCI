import os
import torch
import torch.nn as nn
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
from sklearn.model_selection import train_test_split
import time
import json


def create_data_loaders(
    dataset: geom_data.Dataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[geom_loader.DataLoader, geom_loader.DataLoader, geom_loader.DataLoader]:
    """
    Create train, validation, and test data loaders from a dataset.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        stratify: Whether to stratify the split based on labels
        
    Returns:
        Train, validation, and test data loaders
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get dataset size
    n_samples = len(dataset)
    
    # Get indices for all samples
    indices = list(range(n_samples))
    
    # Prepare stratification if needed
    if stratify:
        try:
            labels = [data.y.item() for data in dataset]
            strat_array = labels
        except:
            print("Warning: Could not extract labels for stratification, using random split instead")
            strat_array = None
    else:
        strat_array = None
    
    # Calculate sizes of each split
    test_size = 1.0 - train_ratio - val_ratio
    
    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=seed,
        stratify=strat_array
    )
    
    # Update stratification array for the second split
    if strat_array:
        strat_array = [strat_array[i] for i in train_val_indices]
    
    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=strat_array
    )
    
    # Create SubsetDataLoader for each split
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = geom_loader.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = geom_loader.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = geom_loader.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def setup_trainer(
    experiment_name: str,
    log_dir: str = "logs",
    max_epochs: int = 100,
    gpus: int = 1,
    precision: int = 16,
    early_stopping_patience: int = 10,
    monitor_metric: str = "val_auc",
    monitor_mode: str = "max",
    save_top_k: int = 3,
    log_every_n_steps: int = 50,
    deterministic: bool = True,
    benchmark: bool = True,
    additional_callbacks: List = None
) -> Tuple[pl.Trainer, str]:
    """
    Set up a PyTorch Lightning trainer with standard callbacks.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
        max_epochs: Maximum number of epochs to train
        gpus: Number of GPUs to use (0 for CPU)
        precision: Precision for training (16, 32, etc.)
        early_stopping_patience: Number of epochs to wait before early stopping
        monitor_metric: Metric to monitor for early stopping and checkpoints
        monitor_mode: Mode for monitoring ("min" or "max")
        save_top_k: Number of best checkpoints to save
        log_every_n_steps: Frequency of logging
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to use cudnn benchmarking
        additional_callbacks: List of additional callbacks
        
    Returns:
        Configured trainer and checkpoint directory
    """
    # Set timestamp for unique run ID
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{experiment_name}-{timestamp}"
    
    # Set up logging
    tensorboard_logger = TensorBoardLogger(log_dir, name=experiment_name, version=timestamp)
    csv_logger = CSVLogger(log_dir, name=experiment_name, version=timestamp)
    
    # Define checkpoint directory
    checkpoint_dir = os.path.join(log_dir, experiment_name, timestamp, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create standard callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        mode=monitor_mode,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Combine all callbacks
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    # Configure trainer
    accelerator = "gpu" if gpus > 0 else "cpu"
    devices = gpus if gpus > 0 else None
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=[tensorboard_logger, csv_logger],
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        benchmark=benchmark
    )
    
    return trainer, checkpoint_dir


def train_model(
    model_class,
    model_kwargs: Dict,
    dataset,
    experiment_name: str,
    batch_size: int = 256,
    train_kwargs: Dict = None,
    save_config: bool = True
) -> Tuple[pl.LightningModule, Dict]:
    """
    Train a model using PyTorch Lightning.
    
    Args:
        model_class: PyTorch Lightning model class
        model_kwargs: Keyword arguments for model initialization
        dataset: Dataset to train on
        experiment_name: Name of the experiment
        batch_size: Batch size for training
        train_kwargs: Additional training parameters
        save_config: Whether to save configuration
        
    Returns:
        Trained model and test metrics
    """
    # Default training parameters
    default_train_kwargs = {
        "max_epochs": 100,
        "gpus": torch.cuda.device_count(),
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "num_workers": min(os.cpu_count(), 8),
        "early_stopping_patience": 10,
        "monitor_metric": "val_auc",
        "monitor_mode": "max",
        "log_dir": "logs"
    }
    
    # Update with provided parameters
    if train_kwargs:
        default_train_kwargs.update(train_kwargs)
    
    train_kwargs = default_train_kwargs
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        train_ratio=train_kwargs["train_ratio"],
        val_ratio=train_kwargs["val_ratio"],
        batch_size=batch_size,
        num_workers=train_kwargs["num_workers"]
    )
    
    # Create model
    model = model_class(**model_kwargs)
    
    # Set up trainer
    trainer, checkpoint_dir = setup_trainer(
        experiment_name=experiment_name,
        log_dir=train_kwargs["log_dir"],
        max_epochs=train_kwargs["max_epochs"],
        gpus=train_kwargs["gpus"],
        early_stopping_patience=train_kwargs["early_stopping_patience"],
        monitor_metric=train_kwargs["monitor_metric"],
        monitor_mode=train_kwargs["monitor_mode"]
    )
    
    # Save configuration if requested
    if save_config:
        config = {
            "model_params": model_kwargs,
            "training_params": train_kwargs,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        model = model_class.load_from_checkpoint(best_model_path)
    
    # Test model
    test_results = trainer.test(model, test_loader)
    
    # Save test results
    results_path = os.path.join(os.path.dirname(checkpoint_dir), "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results[0], f, indent=4)
    
    return model, test_results[0]


def evaluate_model(
    model: pl.LightningModule,
    dataset: geom_data.Dataset,
    batch_size: int = 256,
    num_workers: int = 4
) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained PyTorch Lightning model
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        
    Returns:
        Evaluation metrics
    """
    # Create data loader
    loader = geom_loader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Set up trainer for testing only
    trainer = pl.Trainer(accelerator="auto", devices=1)
    
    # Test model
    test_results = trainer.test(model, loader)
    
    return test_results[0]