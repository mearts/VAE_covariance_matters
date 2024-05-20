import argparse
import torch
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
from utils.utils import (
    calculate_dihedrals,
    calculate_bond_angles,
    calculate_bond_lengths,  
    get_device,
    process_data,
)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device, use_gpu = get_device()

parser = argparse.ArgumentParser(description='Generate_GT_plots')
parser.add_argument('--data_file_path', type=str, default = "./data/df_MD_1pga.pickle", help='Path to data file (pandas dataframe, .pickle), only needed for in-house simulation (1pga)')
parser.add_argument('--protein', type=str, choices=['1unc', '1fsd', '1pga', 'chig', '2f4k'], help='Protein name')
parser.add_argument('--pdb_file_path', type=str, help='Path to pdb file for this protein')

args = parser.parse_args()

coords = process_data(args)['coords']

pwd = torch.norm(coords[:, None, :, :] - coords[:, :, None, :], dim=-1)
d_mean = pwd.mean(dim=0)
d_std = pwd.std(dim=0)

di = calculate_dihedrals(coords)
ba = calculate_bond_angles(coords)
bl = calculate_bond_lengths(coords)


fig, ax = plt.subplots(3, 2, figsize=(12, 18), facecolor = 'white')
ax = ax.flatten()

ax[0].hist(bl[:, 0::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$N-C_{\alpha}$')
ax[0].hist(bl[:, 1::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$C_{\alpha}-C$')
ax[0].hist(bl[:, 2::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$C_{\alpha}-N$')
ax[0].set_title("\nBond lengths", fontsize = 20, y = 1.04)
ax[0].set_xlabel("Length [pm]", fontsize=18, labelpad=10)
ax[0].set_ylabel("Density", fontsize=18, labelpad=10)
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[0].legend(fontsize=18)

ax[1].hist(ba[:, 0::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_{Ca}$')
ax[1].hist(ba[:, 1::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_C$')
ax[1].hist(ba[:, 2::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_N$')
ax[1].set_title("\nBond angles", fontsize = 20, y = 1.04)
ax[1].set_xlabel("Angle", fontsize=18, labelpad=10)
ax[1].set_ylabel("Density", fontsize=18, labelpad=10)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[1].legend(fontsize=18)

ax[2].hist(di[:, 0::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\psi$')
ax[2].hist(di[:, 1::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\omega$')
ax[2].hist(di[:, 2::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\phi$')
ax[2].set_title("\nDihedrals", fontsize = 20, y = 1.04)
ax[2].set_xlabel("Angle", fontsize=18, labelpad=10)
ax[2].set_ylabel("Density", fontsize=18, labelpad=10)
ax[2].set_xlim(-np.pi-0.2, np.pi+0.2)
ax[2].set_xticks([-3,-2,-1,0,1,2,3])
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[2].legend(fontsize=18)

sns.scatterplot(x=(di[:, 2::3][:, 1:-1]).flatten(), y=(di[:, 0::3][:, 1:-1]).flatten(), alpha = 0.1, linewidth=0,  ax=ax[3], label="Samples", legend=False)
ax[3].set_title("\nRamachandran", fontsize = 20, y = 1.04)
ax[3].set_xlabel(r"$\phi$", fontsize = 18, labelpad = 10)
ax[3].set_ylabel(r"$\psi$", fontsize = 18, labelpad = 10)
ax[3].set_xlim(-np.pi-0.2, np.pi+0.2)
ax[3].set_xticks([-3,-2,-1,0,1,2,3])
ax[3].set_yticks([-3,-2,-1,0,1,2,3])
ax[3].set_ylim(-np.pi-0.2, np.pi+0.2)
ax[3].tick_params(axis='both', which='major', labelsize=16)

sns.heatmap(data=d_mean, cbar = True, square = True, annot = False, cmap = 'magma', xticklabels = [], yticklabels = [], ax=ax[4])
ax[4].set_title("\nMean distances", fontsize = 20, pad = 30)
ax[4].set_xlim(0, d_mean.shape[1])
ax[4].set_ylim(0, d_mean.shape[1])

sns.heatmap(data=d_std, cbar = True, square = True, annot = False, cmap = 'magma', xticklabels = [], yticklabels = [], ax=ax[5])
ax[5].set_title("\n"+r"$\sigma$ distances", fontsize = 20, pad = 30)
ax[5].set_xlim(0, d_mean.shape[1])
ax[5].set_ylim(0, d_mean.shape[1])

fig.tight_layout()

fig.savefig(f'../results/fluct/gt_{args.protein}.png', format='png')
