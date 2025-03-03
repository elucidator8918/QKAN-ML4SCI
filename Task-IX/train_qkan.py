import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from models.qkan import QKAN
from models.utils.train import train_hep, train_mnist

# Set manual seed for reproducibility
torch.manual_seed(42)

def train_qkan(task: str):
    """
    Train a KAN model on either the HEP or MNIST dataset.
    
    Args:
        task (str): The dataset to train on, either "HEP" or "MNIST".
    """
    if task not in ["HEP", "MNIST"]:
        raise ValueError("Invalid task. Choose either 'HEP' or 'MNIST'.")

    wandb.init(project="kan_model_training", name=f"{task.lower()}_quantum_experiment")

    if task == "HEP":
        from data.hls4ml import get_dataloaders

        # Model configuration for HEP
        qkan_model = QKAN(layers_hidden=[53, 32, 16, 5], grid_size=7, spline_order=13) 
        trainloader, valloader = get_dataloaders(
            dataset_path="hls4ml-lhc-jet-dataset", train_max=61, val_max=27, batch_size=2048
        )
        train_func = train_hep

    else:  # task == "MNIST"
        from data.mnist import get_dataloaders

        # Model configuration for MNIST
        qkan_model = QKAN(layers_hidden=[28*28, 64, 10], grid_size=7, spline_order=13)
        trainloader, valloader = get_dataloaders(dataset_path="./MNIST", batch_size=64)
        train_func = train_mnist

    # Define optimizer and loss function
    optimizer = optim.AdamW(qkan_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_func(qkan_model, trainloader, valloader, criterion, optimizer, num_epochs=10)

train_qkan(task="HEP")  # Change to "MNIST" if needed