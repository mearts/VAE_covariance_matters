"""
This script can be used to evaluate a model.
Config files can be found in the wandb directory, 
also printed on top of the wandb log.
The path should be of the form:
wandb/run-<date>_<time>-<id>/files/config.yaml
"""

import torch
import argparse
import os
import yaml

from models_and_trainer.VAE_model import VAE
from models_and_trainer.trainer import Trainer
from utils.utils import (
    initialize,
    process_data,
)

############### Arguments ###############

parser = argparse.ArgumentParser(description='VAE_dynamics_resume_training')

parser.add_argument('--config_file_path', type=str, help='Path to config yaml file')
parser.add_argument('--wandb_user', type=str, default='marts', help='Wandb user name')
parser.add_argument('--wandb_project', type=str, default='VAE_dynamics', help='Wandb project name')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
parser.add_argument('--num_samples_z_eval', type=int, default=None, help='Number of latent space samples')

args = parser.parse_args()

with open(args.config_file_path, "r") as f:
    config=yaml.safe_load(f)
for (key, value) in config.items():
    if not "wandb" in key and key != 'batch_size':
        setattr(args, key, value['value'])

if args.num_samples_z_eval is None:
    args.num_samples_z_eval = args.num_samples_z

args.epochs = 0

############### Inintializing ###############

device, use_gpu, save_folder = initialize(args)

############### Data ###############

data_dict = process_data(args)
data_dict['kappa_train'], data_dict['coords_train'] = None, None
data_dict[data_dict['kappa_val']], data_dict['coords_val'] = None, None

############### Model ###############

vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
vae.mean_only = False
opt = None
print(f"Model: {args.model_name}")
print(vae)

############### Evaluation ###############

trainer = Trainer(vae, args, data_dict, save_folder, eval_only=True)

# Load checkpoint
checkpoint = torch.load(trainer.save_loc, map_location=torch.device('cpu'))
trainer.vae.load_state_dict(checkpoint['model_state_dict'])

trainer.get_plots(data_dict['kappa'], data_dict['coords'], data_dict['coords_pNeRF'], 
                  num_samples_z=args.num_samples_z_eval, batch_size=args.batch_size, 
                  topology=data_dict['top'], save=os.path.join(save_folder, args.model_name + "_samples.png"), 
                  model_name=args.model_name)