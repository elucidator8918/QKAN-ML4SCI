# Quantum Kolmogorov-Arnold Networks for High Energy Physics Analysis at the LHC
<img src="https://github.com/user-attachments/assets/a1def6b6-d717-47a9-a95f-edb010b8b966" />

## Description
The ambitious HL-LHC program will require enormous computing resources in the next two decades. New technologies are being sought after to replace the present computing infrastructure. A burning question is whether quantum computer can solve the ever growing demand of computing resources in High Energy Physics (HEP) in general and physics at LHC in particular.

Discovery of new physics requires the identification of rare signals against immense backgrounds. Development of machine learning methods will greatly enhance our ability to achieve this objective. With this project we seek to implement Quantum Machine Learning methods for LHC HEP analysis based on the Pennylane framework. This will enhance the ability of the HEP community to use Quantum Machine Learning methods.

## High-pT Jets Dataset - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3601436.svg)](https://doi.org/10.5281/zenodo.3601436)

This dataset consists of high-pT jets from simulations of LHC proton-proton collisions. It is prepared for [FastML/HLS4ML](https://fastmachinelearning.org) studies

- **High-level features**: Refer to the paper [arXiv:1804.06913](https://arxiv.org/abs/1804.06913) for details.
- **Jet images**: Up to 30 particles per jet, as described in [arXiv:1908.05318](https://arxiv.org/abs/1908.05318).
- **Feature list**: List of jet features with up to 30 particles per jet, see [arXiv:1908.05318](https://arxiv.org/abs/1908.05318).

## Results
### Experiment Comparisons
I conducted two experiments—one using a classical approach and another leveraging quantum machine learning. The results are summarized below:

#### **Classical Experiment**
- **Training Accuracy**: 83.37%
- **Training Loss**: 0.459
- **Validation Accuracy**: 82.83%
- **Validation Loss**: 0.498

#### **Quantum Experiment**
- **Training Accuracy**: 82.05%
- **Training Loss**: 0.512
- **Validation Accuracy**: 82.75%
- **Validation Loss**: 0.503

The classical approach achieved slightly higher training accuracy, while the quantum approach has slightly higher validation accuracy hinting to better generalizations found via the Quantum KAN approach compared to Classical KAN.

[🔗 View Report on Weights & Biases](https://api.wandb.ai/links/sololicht/tszejbc4)
