"""
This script can be used to start training from the last saved checkpoint
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
parser.add_argument('--epochs', type=int, default=1000, help='Number of additional epochs')
parser.add_argument('--wandb_user', type=str, default='marts', help='Wandb user name')
parser.add_argument('--wandb_project', type=str, default='VAE_dynamics', help='Wandb project name')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')

args = parser.parse_args()

with open(args.config_file_path, "r") as f:
    config=yaml.safe_load(f)
for (key, value) in config.items():
    if (not "wandb" in key) and (key != "epochs"):
        setattr(args, key, value['value'])

############### Inintializing ###############

device, use_gpu, save_folder = initialize(args)

############### Data ###############

data_dict = process_data(args)

############### Model ###############

vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
print(f"Model: {args.model_name}")
print(vae)

############### Training ###############

# Train from checkpoint
save_loc = os.path.join(save_folder, args.model_name + ".pt")
checkpoint = torch.load(save_loc, map_location=torch.device(device))
trainer = Trainer(vae, args, data_dict, save_folder, start_epoch=checkpoint['epoch'])
trainer.vae.load_state_dict(checkpoint['model_state_dict'])
trainer.opt.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.train()

# Evaluate (best model)
with torch.no_grad():
    trainer.opt=None
    args.predict_prior = False
    trainer.vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
    trainer.vae.mean_only = False
    checkpoint = torch.load(trainer.save_loc, map_location=torch.device('cpu'))
    trainer.vae.load_state_dict(checkpoint['model_state_dict'])
    trainer.vae = trainer.vae.to(device)

trainer.get_plots(data_dict['kappa'], data_dict['coords'], data_dict['coords_pNeRF'], 
                  num_samples_z=args.num_samples_z, batch_size=args.batch_size, 
                  topology=data_dict['top'], save=os.path.join(save_folder, args.model_name + "_samples.png"), 
                  model_name=args.model_name)