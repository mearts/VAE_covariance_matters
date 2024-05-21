# Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters

This repository corresponds to the paper [Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters](https://arxiv.org/abs/2302.13711).

> [!WARNING]  
> This repository is still under development. We are working on more elaborate documention with extra examples.

## Approach
This model generates protein structure fluctuations by sampling from a distribution over **dihedrals** and **bond angles** (bond lengths fixed), while still respecting constraints in **Euclidean space**. 

<img src="./images/fig1.png" alt="model" width="340"/>

## Data


## Model architecture
The architecture for the proof-of-concept VAE model introduced in the paper is show below.

<img src="./images/model.png" alt="model" width="680"/>

## How to run

### Training

### Config files
Logging of training curves and figures is done in wandb.

#### Resuming or extending training

### Evaluation


## Cite
Please cite the following paper when using this code base:

MLA:
```
Arts, Marloes, Jes Frellsen, and Wouter Boomsma. "Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters." arXiv preprint arXiv:2302 13711 (2023).
```

<br>

BibTeX:
```bibtex
@article{arts2023internal,
  title={Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters},
  author={Arts, Marloes and Frellsen, Jes and Boomsma, Wouter},
  journal={arXiv preprint arXiv:2302.13711},
  year={2023}
  }
```