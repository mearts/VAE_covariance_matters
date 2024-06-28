import sys
import numpy as np
import torch
from deeptime.decomposition import TICA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import argparse
from utils.tica_utils import Args, get_tic_features

sys.path.insert(0, "../")
from utils.utils import process_data

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='GT_TICA')
parser.add_argument('--config_file_path', type=str, help='Path to config file', required=True)
parser.add_argument('--num_samples_z', type=int, default=400000, help='Number of samples to draw from latent space (z)')
parser.add_argument('--bs', type=int, default=100000, help='Batch size for feature extraction')
parser.add_argument('--nbins', type=int, default=101, help='Number of bins for histogram')
parser.add_argument('--partial_fit', type=bool, default=False, help='Do partial fit')
parser.add_argument('--only_save_probs', action='store_true', help='Only save probs, no plotting or model saving')

 
args_new = parser.parse_args()

############################################################################################

# LOADING DATA
print("Loading data...", flush=True)
path_to_main = '../'

args = Args(args_new.config_file_path, path_to_main)
data_dict = process_data(args)


############################################################################################

# FIT TICA
print("Fitting TIC model", flush=True)
top = data_dict['top']
top_tica = {}

top_tica_ind = top["topology"].select('name N or name CA or name C')
top_tica["topology"] = top["topology"].subset(top_tica_ind)
if top["order"] is not None:
    top_tica["order"] = top["order"][top_tica_ind]
else:
    top_tica["order"] = None

tic_features = get_tic_features(data_dict['coords'], top_tica, args_new.bs)
print("Features extracted", flush=True)

tica = TICA(lagtime=100, dim=2)

if args_new.partial_fit:
    raise NotImplementedError("Partial fit not yet implemented (timeshifted_split not in version on cluster)")
else:
    tica.fit(tic_features)
    tica = tica.fetch_model()
    transformed_data = tica.transform(tic_features)

prob_gt, bin_edges_x, bin_edges_y = np.histogram2d(transformed_data[:, 0], transformed_data[:, 1], bins=args_new.nbins, density=True)
bin_mids_x = (bin_edges_x[1:] + bin_edges_x[:-1]) / 2
bin_mids_y = (bin_edges_y[1:] + bin_edges_y[:-1]) / 2


############################################################################################

if args_new.only_save_probs:
    save_name = f"./TICA_models/probs_TICA_{args.protein}_bb.npy"
    np.save(save_name, -np.log(prob_gt.T))

else:

    # PLOT

    cmap = 'inferno_r'

    fig, ax = plt.subplots(figsize=(8,8))
    plot=ax.imshow(-np.log(prob_gt.T), origin="lower", cmap=cmap, vmin=-2, vmax=8)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("TIC 0", fontsize = 18, labelpad = 10)
    ax.set_ylabel("TIC 1", fontsize = 18, labelpad = 10)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.2)
    plt.colorbar(plot, cax=cax)
    cax.set_ylabel(r"Free energy / $k_BT$", fontsize = 18, labelpad=10)
    cax.tick_params(labelsize=16)

    plt.tight_layout()

    plt.savefig(f'../results/TICA/TICA_MD_{args.protein}_bb.png', format = 'png')
    plt.savefig(f'../results/TICA/TICA_MD_{args.protein}_bb.pdf', format = 'pdf')
    print("GT (MD) figures saved", flush=True)


    ############################################################################################

    # SAVE TICA MODEL

    save_name = f"./TICA_models/fitted_TICA_{args.protein}_bb.pickle"

    with open(save_name, 'wb') as pickle_file:
        pickle.dump({'tica': tica,
                    'bin_edges_x': bin_edges_x,
                    'bin_edges_y': bin_edges_y,
                    'top': top_tica}, pickle_file)

    print("TICA model saved", flush=True)
