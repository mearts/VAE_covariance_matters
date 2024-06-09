# Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters

This repository corresponds to the paper [Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters](https://openreview.net/forum?id=9XRZtZRmEB).

## Approach
This model generates protein structure fluctuations by sampling from a distribution over **dihedrals** and **bond angles** (bond lengths fixed), while still respecting constraints in **Euclidean space**. 

<img src="./images/fig1.png" alt="model" width="400"/>

## Requirements
Dependencies are listed in [VAE_covariane_matters.yml](./VAE_covariane_matters.yml), which can also be used to create a new conda environment:
```
conda env create -f VAE_covariane_matters.yml
```
> [!NOTE]
> 1. You might have to change the versions of pytorch and/or cudatoolkit version based on the system you're running on.
> 2. In case you're running on cpu, comment out the installation of cudatoolkit.

## Data
Our paper considers three types of datasets:
* In-house MD simulation of Protein G (1pga). File [data/df_MD_1pga.npy](data/df_MD_1pga.npy) contains an array with coordinates for the simulation, with corresponding pdb file [data/1pga.pdb](data/1pga.pdb).
* NMR datasets for the human villin headpiece (1unc) and BBA motif (1fsd). These sets are taken directly from the Protein Data Bank and can be found at [data/1unc.pdb](data/1unc.pdb) and [data/1fsd.pdb](data/1fsd.pdb), respectively.
* Fast-folder simulations of chignolin (cln025) and the chicken villin headpiece (2f4k). These datasets are available upon request from the authors of [Lindorff-Larsen et al. (2011)](https://www.science.org/doi/full/10.1126/science.1208351).

## Model architecture
The architecture for the proof-of-concept VAE model introduced in the paper is show below. The main components are a simple MLP-based VAE and a U-Net that predicts Lagrange multipliers as a proxy for atom fluctuations. See [models_and_trainer](./models_and_trainer/) for implementation.

<img src="./images/model.png" alt="model" width="680"/>

## How to run

> [!IMPORTANT] 
> Logging of training, evaluation metrics, images and run configurations is done using Weights & Biases (wandb). While it is possible to run training without wandb using the `--no_wandb` flag, resuming training and doing evaluation requires a config file. Therefore, to avoid having to construct config files manually, we strongly recommend using wandb, since the config file is then directly available in the wandb directory of the training run. This directory is also printed at the initialization of each run, so it can easily be retrieved.

### Training
#### Running the VAE with contraints


#### Running baselines
Also include flow here

#### Resuming or extending training


### Evaluation
#### Evaluating the VAE with constraints

#### Evaluating baselines

#### TICA analysis


## Cite
Please cite the following paper when using this code base:

```bibtex
@article{
arts2024internalcoordinate,
title={Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters},
author={Marloes Arts and Jes Frellsen and Wouter Boomsma},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=9XRZtZRmEB},
note={}
}
```