import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from models.kan import KAN
from data.hls4ml import get_dataloaders
from models.utils.train import train

torch.manual_seed(42)
# Create a KAN model: 53D inputs, 5D output, with 32, 16, 5 hidden neurons.
# Uses cubic spline (k=13) and 7 grid intervals.
kan_model = KAN(layers_hidden=[53, 32, 16, 5, 5], grid_size=7, spline_order=13)
wandb.init(project="kan_model_training", name="classical_experiment")

# Define optimizer and loss function
optimizer = optim.AdamW(kan_model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

trainloader, valloader = get_dataloaders(dataset_path="hls4ml-lhc-jet-dataset", train_max=61, val_max=27, batch_size=2048)
train(kan_model, trainloader, valloader, criterion, optimizer, num_epochs=10)