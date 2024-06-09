"""
Main training script
"""

import torch
import argparse
import os

from models_and_trainer.VAE_model import VAE
from models_and_trainer.trainer import Trainer
from utils.utils import (
    process_args,
    initialize,
    process_data, 
)

############### Arguments ###############

parser = argparse.ArgumentParser(description='VAE_dynamics_main')

parser.add_argument('--save_folder', type=str, default='./results', help='Folder to save results')
parser.add_argument('--model_name', type=str, help='Supply name for current model')
parser.add_argument('--data_file_path', type=str, default = "./data/df_MD_1pga.npy", help='Path to MD coordinates file (numpy array saved as .npy), only needed for in-house simulation (1pga)')
parser.add_argument('--protein', type=str, choices=['1unc', '1fsd', '1pga', 'chig', '2f4k'], help='Protein name')
parser.add_argument('--pdb_file_path', type=str, help='Path to pdb file for this protein. In case of NMR data, corresponds to full data set')

parser.add_argument('--wandb_user', type=str, help='Wandb user name')
parser.add_argument('--wandb_project', type=str, help='Wandb project name')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')

parser.add_argument('--latent_features', type=int, default=16, help='Number of latent features')
parser.add_argument('--encoder_list', nargs="+", type=int, default=[128, 64, 32], help='Encoder sizes (large to small')
parser.add_argument('--decoder_list', nargs="+", type=int, default=[32, 64, 128], help='Decoder sizes (small to large')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_warm_up_KL', type=int, default=200, help='Number of epochs to warm up KL-divergence in ELBO')
parser.add_argument('--num_mean_only', type=int, default=100, help='Number of epochs to train only mean kappa')
parser.add_argument('--num_samples_z', type=int, default=100, help='Number of latent space samples')
parser.add_argument('--a_weight', type=float, default=25., help='Starting value for precomputed prior scaling factor a')
parser.add_argument('--lambda_aux_weight', type=float, default=25, help='Starting value weight on auxiliary loss (MAE on inverse lambda)')

parser.add_argument('--predict_prior', action='store_true', help='Predict prior: baseline')
parser.add_argument('--constraints_off', action='store_true', help='Turn off constraints (covariance based only on the prior): baseline')
parser.add_argument('--constraints_only', action='store_true', help='Construct covariance based only on constraints (ignore prior): baseline')


args = parser.parse_args()
args = process_args(args)

############### Initializing ###############

device, use_gpu, save_folder = initialize(args)

############### Data ###############

data_dict = process_data(args)

############### Model ###############

# Model
vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
print(f"Model: {args.model_name}")
print(vae)

############### Training ###############

# Train
trainer = Trainer(vae, args, data_dict, save_folder)
trainer.train()

# Evaluate (best model)
with torch.no_grad():
    trainer.opt=None
    trainer.vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
    trainer.vae.mean_only = False
    checkpoint = torch.load(trainer.save_loc, map_location=torch.device('cpu'))
    trainer.vae.load_state_dict(checkpoint['model_state_dict'])
    trainer.vae = trainer.vae.to(device)

trainer.get_plots(data_dict['kappa'], data_dict['coords'], data_dict['coords_pNeRF'],
                  num_samples_z=args.num_samples_z, batch_size=args.batch_size, 
                  topology=data_dict['top'], save=os.path.join(save_folder, args.model_name + "_samples.png"), 
                  model_name=args.model_name)