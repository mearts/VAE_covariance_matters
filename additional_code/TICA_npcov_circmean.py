import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import circmean
from sklearn.covariance import MinCovDet, empirical_covariance, oas
import argparse
from utils.additional_utils import Args, get_tic_features
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import sys
sys.path.insert(0, "../")
from utils.utils import process_data, prepare_for_pnerf, samples_to_structures


parser = argparse.ArgumentParser(description='TICA_npcov_circvar')
parser.add_argument('--config_file_path', type=str, help='Path to config file', required=True)
parser.add_argument('--num_samples_z', type=int, default=400000, help='Number of samples to draw from latent space (z)')
parser.add_argument('--bs', type=int, default=100000, help='Batch size for feature extraction')
parser.add_argument('--only_save_probs', action='store_true', help='Only save probs, no plotting or model saving')
parser.add_argument('--seed_off', action='store_true', help="Don't use seed for sampling")
parser.add_argument('--label', type=str, default="", help='Label for saving')

args_new = parser.parse_args()

if not args_new.seed_off:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

args_new.label = args_new.label if len(args_new.label) == 0 else f"_{args_new.label}"

############################################################################################

# LOAD DATA AND GET CIRCMEAN
print("Loading data...", flush=True)

path_to_main = '../'
args = Args(args_new.config_file_path, path_to_main)

saved_model = f"./TICA_models/fitted_TICA_{args.protein}_bb.pickle"
assert os.path.isfile(saved_model), f"Missing saved TICA model for {args.protein}_bb"

data_dict = process_data(args)
kappa, bond_lengths = data_dict['kappa'], data_dict['bond_lengths_pNeRF']

save_circmean = f"./circmeans/{args.protein}_circmean.pt"

if os.path.isfile(save_circmean):
    kappa_mean = torch.load(save_circmean)
else:
    kappa_mean = torch.from_numpy(circmean(kappa, axis=0, low=-np.pi, high=np.pi))
    torch.save(kappa_mean, save_circmean)

############################################################################################

# SAMPLE USING ESTIMATOR PRECISION AND TRANSFORM
print("Transforming standard estimator samples...", flush=True)

num_dihedrals = (len(kappa_mean) - 1) // 2
kappa_mean = kappa_mean.unsqueeze(dim=0)
kappa_offsets = torch.atan2(torch.sin(kappa - kappa_mean), torch.cos(kappa - kappa_mean))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    kappa_npcov = torch.from_numpy(empirical_covariance(kappa_offsets)).float().to(device)
    MN = MultivariateNormal(torch.zeros_like(kappa[0], device=device), covariance_matrix=kappa_npcov)
    covlabel = 'empirical'
except ValueError:
    try:
        kappa_npcov = torch.from_numpy(MinCovDet(random_state=42).fit(kappa_offsets).covariance_).float().to(device)
        MN = MultivariateNormal(torch.zeros_like(kappa[0], device=device), covariance_matrix=kappa_npcov)
        covlabel = 'robust, MinCovDet'
    except ValueError:
        kappa_npcov = torch.from_numpy(oas(kappa_offsets, assume_centered=True)[0]).float().to(device)
        MN = MultivariateNormal(torch.zeros_like(kappa[0], device=device), covariance_matrix=kappa_npcov)
        covlabel = 'shrunk, OAS'

npcov_samples = MN.sample(torch.Size([args_new.num_samples_z])) + kappa_mean.to(device)
npcov_samples = torch.atan2(torch.sin(npcov_samples), torch.cos(npcov_samples))
dih_npcov, ba_npcov = npcov_samples[:, :num_dihedrals], npcov_samples[:, num_dihedrals:]

di_pNeRF = prepare_for_pnerf(dih_npcov, kappa_type="di")
ba_pNeRF = prepare_for_pnerf(ba_npcov, kappa_type="ba")
bond_lengths = bond_lengths.repeat(1, args_new.num_samples_z, 1).to(device)
npcov_coord = samples_to_structures(di_pNeRF, bond_lengths, ba_pNeRF).cpu()

with open(saved_model, 'rb') as pickle_file:
    tica_dict = pickle.load(pickle_file)

tic_features_samp = get_tic_features(npcov_coord, tica_dict['top'], args_new.bs)
transformed_samp = tica_dict['tica'].transform(tic_features_samp)

prob_samp, _, _ = np.histogram2d(transformed_samp[:, 0], transformed_samp[:, 1], bins=[tica_dict['bin_edges_x'], 
                    tica_dict['bin_edges_y']], density=True)


############################################################################################

if args_new.only_save_probs:
    save_name = f"./TICA_models/probs_TICA_npcov_{args.protein}{args_new.label}_bb.npy"
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

    plt.savefig(f'../results/TICA/TICA_npcov_{args.protein}{args_new.label}_bb.png', format = 'png')
    print("Standard estimator figures saved", flush=True)
