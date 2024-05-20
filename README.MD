# Internal-Coordinate Density Modelling of Protein Structure: Covariance Matters

## Approach
This model generates protein dynamics structures by sampling from a distribution over **dihedrals** and **bond angles** (bond lengths fixed), which from now will together be called $\kappa$, while still respecting constraints in **Euclidean space**. A VAE predicts a mean over $\kappa$ conditioned on $z$ and then we construct a full precision matrix over dihedrals and bond angles, which is composed of:
* A precision prior $\Sigma^{-1}_{prior}$, weighed by a tunable parameter $a$, based on the fluctuations over $\kappa$ in the dataset.
* A precision matrix over the constraints $\Sigma^{-1}_{constr}$ we're imposing, i.e. how much points are allowed to fluctuate radially in Euclidean space. The contributions of each point fluctuation are weighed by Lagrange multipliers $\lambda$ that are predicted by a U-Net. 

## Model architecture
<img src="./images/model.png" alt="model" width="680"/>

## Repository contents
* `data/`: a folder containing dynamics data and (optionally) pdb files.
* `dihedrals_bondangles.ipynb`: a Jupyter Notebook to play around with precision matrices in a ground truth setting (without any model).
* `main_VAE_dynamics.py`: main training script. Run `python main_VAE_dynamics.py --help` for options.
* `resume_training.py`: resume training from last saved checkpoint. Run `python resume_training.py --help` for options.
* `eval_VAE_dynamics.py`: evaluate model. Run `python eval_VAE_dynamics.py --help` for options.
* `pnerf_specify.py`: code used to build structures from internal coordinate sequences.
* `trainer_VAE_dynamics.py`: containing the trainer class.
* `unet_model.py`: containing the U-Net model framework (based on https://github.com/milesial/Pytorch-UNet).
* `utils.py`: misscelaneous functions.
* `VAE_model.py`: containing the VAE framework.

## How to run
The model can be trained by running `python main_VAE_dynamics.py <--arguments>`, check `--help` for options. Logging of training curves and figures is done in wandb, and the final model is saved in the specified folder. 
