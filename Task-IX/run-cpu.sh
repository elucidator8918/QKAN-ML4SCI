#!/bin/bash

pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install wandb pennylane numpy pandas matplotlib h5py seaborn scikit-learn kaggle -q
kaggle datasets download aleespinosa/hls4ml-lhc-jet-dataset
unzip hls4ml-lhc-jet-dataset.zip -d hls4ml-lhc-jet-dataset
rm -rf hls4ml-lhc-jet-dataset.zip
wandb login e459fcecd8152b93378191a85367ad88e8b67f3f