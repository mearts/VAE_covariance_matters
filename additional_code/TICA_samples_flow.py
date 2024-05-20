import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from utils.additional_utils import get_tic_features
import argparse


parser = argparse.ArgumentParser(description='TICA_flow_samples')
parser.add_argument('--samples_path', type=str, help='Path to flow samples', required=True)
parser.add_argument('--protein', type=str, default='protein', help='Protein name')
parser.add_argument('--bs', type=int, default=0, help='Batch size for getting tic features')
parser.add_argument('--only_save_probs', action='store_true', help='Only save probs, no plotting or model saving')
parser.add_argument('--seed_off', action='store_true', help="Don't use seed for sampling")

args = parser.parse_args()

if not args.seed_off:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

############################################################################################

# Load TICA model
saved_model = f"./TICA_models/fitted_TICA_{args.protein[:4]}_bb.pickle"
with open(saved_model, 'rb') as pickle_file:
    tica_dict = pickle.load(pickle_file)

# Get coordinates
samp_coord = torch.from_numpy(np.load(args.samples_path)["samples"]) # in nm
samp_coord = (samp_coord * 10).reshape((samp_coord.shape[0], samp_coord.shape[1]//3, 3))
if tica_dict['top']['order'] is not None:
    samp_coord = samp_coord[:, np.argsort(tica_dict['top']['order']), :] # Put into the right order

print("Flow samples loaded", flush=True)

############################################################################################

# Run TICA
bs = samp_coord.shape[0] if args.bs==0 else args.bs
tic_features = get_tic_features(samp_coord, tica_dict['top'], args.bs)
del samp_coord

transformed_samp = tica_dict['tica'].transform(tic_features)
del tic_features

prob_samp, _, _ = np.histogram2d(transformed_samp[:, 0], transformed_samp[:, 1], bins=[tica_dict['bin_edges_x'], 
                tica_dict['bin_edges_y']], density=True)

print("TICA done!", flush=True)

############################################################################################

if args.only_save_probs:
    save_name = f"./TICA_models/probs_TICA_flow_{args.protein}.npy"
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

    plt.savefig(f'../results/TICA/TICA_flow_{args.protein}_bb.png', format = 'png')
    print("Flow figures saved", flush=True)
