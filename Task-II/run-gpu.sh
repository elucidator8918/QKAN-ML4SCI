#!/bin/bash

pip install torch torch_geometric pytorch_lightning torchmetrics wandb numpy pandas matplotlib seaborn scikit-learn -q
wget https://zenodo.org/record/3164691/files/QG_jets.npz -P data/raw
wandb login e459fcecd8152b93378191a85367ad88e8b67f3f