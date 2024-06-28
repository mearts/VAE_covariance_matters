import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from utils.tica_utils import Args, get_tic_features
import argparse

sys.path.insert(0, "../")
from models_and_trainer.VAE_model import VAE


parser = argparse.ArgumentParser(description='TICA_VAE_samples')
parser.add_argument('--config_file_path', type=str, help='Path to config file', required=True)
parser.add_argument('--num_samples_z', type=int, default=400000, help='Number of samples to draw from latent space (z)')
parser.add_argument('--bs', type=int, default=32, help='Batch size for VAE sampling')
parser.add_argument('--only_save_probs', action='store_true', help='Only save probs, no plotting or model saving')
parser.add_argument('--seed_off', action='store_true', help="Don't use seed for sampling")
parser.add_argument('--label', type=str, default="", help="Extra label to incorporate in save name")

args_new = parser.parse_args()

if not args_new.seed_off:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

############################################################################################

# LOAD MODEL
print("Loading model...", flush=True)
path_to_main = '../'

args = Args(args_new.config_file_path, path_to_main)

# Load checkpoint
save_loc = os.path.join("../models", args.model_name + ".pt")
checkpoint = torch.load(save_loc, map_location=torch.device('cpu'))

# VAE model
vae = VAE(args, checkpoint['bond_lengths'], checkpoint['prior'])
vae.mean_only = False
vae.load_state_dict(checkpoint['model_state_dict'])
vae.to(vae.device)

vae.eval()


############################################################################################

# LOAD TICA MODEL
saved_model = f"./TICA_models/fitted_TICA_{args.protein}_bb.pickle"
with open(saved_model, 'rb') as pickle_file:
    tica_dict = pickle.load(pickle_file)

if not args_new.prior_only:
    # DRAW SAMPLES FROM VAE AND TRANSFORM
    print("Sampling from VAE, extracting features and transforming...", flush=True)

    vae.steps = 1

    num_batches = -(args_new.num_samples_z // -args_new.bs)
    residual = args_new.num_samples_z % args_new.bs

    for b in range(num_batches):
        print(f"Batch [{b+1}/{num_batches}]", flush=True)

        if b == num_batches-1 and residual != 0:
            num_batch_samp = residual
        else:
            num_batch_samp = args_new.bs
        samp_coord = vae.sample(num_batch_samp, topology=None, batch_size=num_batch_samp)['structures']

        tic_features_batch = get_tic_features(samp_coord, tica_dict['top'], num_batch_samp)
        samp_coord = None

        transformed_batch = tica_dict['tica'].transform(tic_features_batch)
        tic_features_batch = None
        transformed_samp = transformed_batch if b==0 else np.vstack((transformed_samp, transformed_batch))
        transformed_batch = None

    prob_samp, _, _ = np.histogram2d(transformed_samp[:, 0], transformed_samp[:, 1], bins=[tica_dict['bin_edges_x'], 
                        tica_dict['bin_edges_y']], density=True)

    print("TICA done!", flush=True)

    ############################################################################################

    if len(args_new.label) > 0:
        args_new.label = f"_{args_new.label}"

    if args_new.only_save_probs:
        save_name = f"./TICA_models/probs_TICA_VAE_{args.protein}_bb_a{int(args.a_weight)}mae{int(args.lambda_aux_weight)}{args_new.label}.npy"
        np.save(save_name, -np.log(prob_samp.T))

    else:

        # PLOT

        fig, ax = plt.subplots(figsize=(8,8))
        plot=ax.imshow(-np.log(prob_samp.T), origin="lower", cmap='inferno_r', vmin=-2, vmax=8)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel("TIC 0", fontsize = 18, labelpad = 10)
        ax.set_ylabel("TIC 1", fontsize = 18, labelpad = 10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.2)
        plt.colorbar(plot, cax=cax)
        cax.set_ylabel(r"Free energy / $k_BT$", fontsize = 18, labelpad=10)
        cax.tick_params(labelsize=16)

        fig.tight_layout()

        plt.savefig(f'../results/TICA/TICA_VAE_{args.protein}_bb_a{int(args.a_weight)}mae{int(args.lambda_aux_weight)}{args_new.label}.png', format = 'png')
        print("VAE figures saved", flush=True)
