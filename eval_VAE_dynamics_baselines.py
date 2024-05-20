"""
This script can be used to evaluate a model
"""

import torch
import argparse
import yaml
import numpy as np
from scipy.stats import circmean
from sklearn.covariance import MinCovDet, empirical_covariance, oas
from torch.distributions import MultivariateNormal
import mdtraj as md
import matplotlib.pyplot as plt
import wandb
import os
from copy import deepcopy
import gc
from datetime import datetime
import pickle

from models_and_trainer.VAE_model import VAE
from models_and_trainer.trainer_VAE_dynamics import Trainer
from utils.utils import (
    initialize,
    process_data,
    prepare_for_pnerf,
    samples_to_structures,
    get_coord_with_O,
)

############### Arguments ###############

parser = argparse.ArgumentParser(description='VAE_dynamics_resume_training')

parser.add_argument('--config_file_path_main', type=str, help='Path to config yaml file, main model')
parser.add_argument('--config_file_path_fix_prior', type=str, default='none', help='Path to config yaml file, fixed prior model')
parser.add_argument('--config_file_path_pred_prior', type=str, default='none', help='Path to config yaml file, predicted prior model')
parser.add_argument('--config_file_path_constr_only', type=str, default='none', help='Path to config yaml file, constraints only model')
parser.add_argument('--flow_sample_path', type=str, default='none', help='Path to flow samples')
parser.add_argument('--wandb_user', type=str, default='marts', help='Wandb user name')
parser.add_argument('--wandb_project', type=str, default='VAE_dynamics_baselines', help='Wandb project name')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
parser.add_argument('--num_samples_z_eval', type=int, default=None, help='Number of latent space samples')
parser.add_argument('--legend_off', action='store_true', help="Don't add a legend to the plot")
parser.add_argument('--seed_off', action='store_true', help="Don't use seed for sampling")
parser.add_argument('--only_save_probs', action='store_true', help='Only save probs, no plotting or model saving')


args_new = parser.parse_args()
args_new.epochs = 0


############### Get results for each model ###############

exp_labels = ["main"]
if args_new.config_file_path_fix_prior != 'none':
    exp_labels.append("fix_prior")
if args_new.config_file_path_pred_prior != 'none':
    exp_labels.append("pred_prior")
if args_new.config_file_path_constr_only != 'none':
    exp_labels.append("constr_only")

all_results = {}
model_counter = 0
for c, config_file_path in enumerate([args_new.config_file_path_main, args_new.config_file_path_fix_prior, \
                         args_new.config_file_path_pred_prior, args_new.config_file_path_constr_only]):

    if config_file_path == 'none':
        continue
    else:
        print(f"Current model: {exp_labels[model_counter]} [{model_counter+1}/{len(exp_labels)}]", flush=True)
        model_counter += 1
    
    ############### Loading ###############

    args = deepcopy(args_new) # TODO: is this the best way to do it?
    with open(config_file_path, "r") as f:
        config=yaml.safe_load(f)
    for (key, value) in config.items():
        if not "wandb" in key and key != 'batch_size':
            setattr(args, key, value['value'])

    if args.num_samples_z_eval is None:
        args.num_samples_z_eval = args.num_samples_z

    if not hasattr(args, 'constraints_off'):
        setattr(args, 'constraints_off', False)
    if not hasattr(args, 'constraints_only'):
        setattr(args, 'constraints_only', False)

    if c == 0: # Save params fo main model for figure save name
        protein = args.protein
        param_a = args.a_start
        param_aux = args.lambda_aux_weight_start

    ############### Inintializing ###############

    device, use_gpu, save_folder = initialize(args)

    ############### Data ###############

    if c == 0: # Only load data once (assumes crop / subsample_data are the same for all models)
        data_dict = process_data(args)
        del data_dict['kappa_train']
        del data_dict['coords_train']
        del data_dict['kappa_val']
        del data_dict['coords_val']

        kappa = data_dict['kappa'].clone()

    ############### Model ###############

    vae = VAE(args, data_dict['bond_lengths_pNeRF'], data_dict['kappa_prior'])
    vae.mean_only = False
    if args.fluctuation_aux == 'cm_prior':
        fluct_prior = data_dict['coords'][data_dict['ind_train'],:,:].var(dim=0).mean(dim=-1)
        vae.fluct_prior = fluct_prior.to(device)
    opt = None
    print(f"Model: {args.model_name}")
    print(f"Auxiliary fluctuation loss: {args.fluctuation_aux}")
    print(f"Auxiliary lambda loss: {args.lambda_aux}")
    print(vae)

    ############### Evaluation ###############

    trainer = Trainer(vae, args, data_dict, save_folder, eval_only=True)

    # Load checkpoint
    checkpoint = torch.load(trainer.save_loc, map_location=torch.device('cpu'))
    trainer.vae.load_state_dict(checkpoint['model_state_dict'])
    
    trainer.vae.eval()

    with torch.no_grad():
        print(f'a: {trainer.vae.a.item()}')
        if args.batch_size == None:
            args.batch_size = len(kappa)

        # Topology
        top2_ind = data_dict['top']["topology"].select('name CA or name N or name C')
        top2 = data_dict['top']["topology"].subset(top2_ind)

        if data_dict['top']["order"] is not None:
            top2_order = data_dict['top']["order"][top2_ind]
            top2_order_reverse = np.argsort(top2_order)
            top2_order = np.argsort(top2_order_reverse)
        else:
            top2_order_reverse, top2_order = None, None

        # Only calculate MD references and standard estimator once
        if c == 0:
            # MD
            if top2_order is not None:
                data_dict['coords'] = data_dict['coords'][:, top2_order, :]
            traj_MD = md.Trajectory(data_dict['coords'].numpy()/10, topology=top2)
            traj_MD = traj_MD.superpose(traj_MD, frame=0).xyz*10
            if top2_order_reverse is not None:
                traj_MD = traj_MD[:, top2_order_reverse, :]
            MD = np.var(traj_MD, axis=0)
            MD_ref = traj_MD[0]

            # np.cov
            kappa_mean = torch.from_numpy(circmean(kappa, axis=0, low=-np.pi, high=np.pi))
            kappa_offsets = torch.atan2(torch.sin(kappa - kappa_mean), torch.cos(kappa - kappa_mean))

            try:
                kappa_npcov = torch.from_numpy(empirical_covariance(kappa_offsets)).float().to(trainer.device)
                MN = MultivariateNormal(torch.zeros_like(kappa[0], device=trainer.device), covariance_matrix=kappa_npcov)
                covlabel = 'empirical'
            except ValueError:
                try:
                    kappa_npcov = torch.from_numpy(MinCovDet(random_state=42).fit(kappa_offsets).covariance_).float().to(trainer.device)
                    MN = MultivariateNormal(torch.zeros_like(kappa[0], device=trainer.device), covariance_matrix=kappa_npcov)
                    covlabel = 'robust, MinCovDet'
                except ValueError:
                    kappa_npcov = torch.from_numpy(oas(kappa_offsets, assume_centered=True)[0]).float().to(trainer.device)
                    MN = MultivariateNormal(torch.zeros_like(kappa[0], device=trainer.device), covariance_matrix=kappa_npcov)
                    covlabel = 'shrunk, OAS'
            del kappa_npcov

            print(f"Standard estimator method: {covlabel}")

            num_samp = args.num_samples_z_eval
            npcov_samples = MN.sample(torch.Size([num_samp])) + kappa_mean.to(trainer.device)
            del kappa_mean
            npcov_samples = torch.atan2(torch.sin(npcov_samples), torch.cos(npcov_samples))
            dih_npcov, ba_npcov = npcov_samples[:, :trainer.vae.num_dihedrals], npcov_samples[:, trainer.vae.num_dihedrals:]

            di_pNeRF = prepare_for_pnerf(dih_npcov, kappa_type="di")
            ba_pNeRF = prepare_for_pnerf(ba_npcov, kappa_type="ba")
            structs_npcov = samples_to_structures(di_pNeRF, trainer.vae.bond_lengths.repeat(1, num_samp, 1), ba_pNeRF).cpu()
            del npcov_samples
            del dih_npcov
            del ba_npcov
            del di_pNeRF
            del ba_pNeRF
            if top2_order is not None:
                structs_npcov = structs_npcov[:, top2_order, :]
            traj_npcov = md.Trajectory(structs_npcov.numpy()/10, topology=top2)
            traj_npcov = traj_npcov.superpose(traj_npcov, frame=0).xyz*10
            if top2_order_reverse is not None:
                traj_npcov = traj_npcov[:, top2_order_reverse, :]
                structs_npcov = structs_npcov[:, top2_order_reverse, :]
            npcov = np.var(traj_npcov, axis=0)

            traj = md.Trajectory(get_coord_with_O(structs_npcov, data_dict['top'])/10, topology=data_dict['top']["topology"])
            model_name = "" if args.model_name is None else args.model_name
            traj.save_pdb("pdb_files/estcov_withO_" + model_name + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".pdb")

        
        # Samples from VAE
        trainer.vae.steps = args.fluctuation_steps
        samples = trainer.vae.sample(args.num_samples_z_eval, data_dict['top'], args.model_name, batch_size=args.batch_size)
                                               
        structs = samples['structures'].clone()
        del samples
        structs = np.concatenate((np.expand_dims(MD_ref, axis=0), structs.numpy()), axis=0)
        if top2_order is not None:
            structs = structs[:, top2_order, :]
        traj_samp = md.Trajectory(structs/10, topology=top2)
        del structs

        samp_aligned = traj_samp.superpose(traj_samp, frame=0).xyz[1:]*10
        if top2_order_reverse is not None:
            samp_aligned = samp_aligned[:, top2_order_reverse, :]
        samp_aligned = np.var(samp_aligned, axis=0)
        all_results[exp_labels[model_counter-1]] = samp_aligned
        del samp_aligned

    del vae
    del trainer
    del args
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Obtained results for model {exp_labels[model_counter-1]} [{model_counter}/{len(exp_labels)}]\n", flush=True)

if args_new.flow_sample_path is not 'none':
    structs = np.load(args_new.flow_sample_path)["samples"] * 10 #nm to angstrom
    structs = structs.reshape((structs.shape[0], structs.shape[1]//3, 3))
    MD_ref = MD_ref if top2_order is None else MD_ref[top2_order, :]
    structs = np.concatenate((np.expand_dims(MD_ref, axis=0), structs), axis=0)
    traj_samp = md.Trajectory(structs/10, topology=top2) # Already in pdb order
    del structs

    samp_aligned = traj_samp.superpose(traj_samp, frame=0).xyz[1:]*10
    if top2_order_reverse is not None:
        samp_aligned = samp_aligned[:, top2_order_reverse, :]
    samp_aligned = np.var(samp_aligned, axis=0)
    all_results['flow'] = samp_aligned
    del samp_aligned


xrange = np.arange(len(MD))

if args_new.only_save_probs:
    prob_dict = {}
    prob_dict["xrange"] = xrange
    prob_dict["Reference"] = np.mean(MD, axis=-1)
    prob_dict["VAE"] = np.mean(all_results["main"], axis=-1)
    prob_dict["npcov"] = np.mean(npcov, axis=-1)
    if args_new.config_file_path_fix_prior != 'none':
        prob_dict["fixedprior"] = np.mean(all_results["fix_prior"], axis=-1)
    if args_new.config_file_path_pred_prior != 'none':
        prob_dict["predprior"] = np.mean(all_results["pred_prior"], axis=-1)
    if args_new.config_file_path_constr_only != 'none':
        prob_dict["constr_only"] = np.mean(all_results["constr_only"], axis=-1)
    if args_new.flow_sample_path != 'none':
        prob_dict["flow"] = np.mean(all_results["flow"], axis=-1)

    save_name = f"./results_{protein}_a{param_a}_mae{param_aux}_fluct_baselines.pickle"
    with open(save_name, "wb") as f:
        pickle.dump(prob_dict, f)
else:
    ############### Plotting ###############
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), facecolor = 'white')
    ax = ax.flatten()

    ax[0].plot(xrange, np.mean(MD, axis=-1), label = 'MD reference', linewidth=3, color='tab:blue')
    ax[0].plot(xrange, np.mean(all_results["main"], axis=-1), label = 'VAE', linewidth=3, color='tab:orange')
    ax[0].plot(xrange, np.mean(npcov, axis=-1), label = 'Standard estimator', linewidth=1.8, linestyle='--', color='#029e73')
    if args_new.config_file_path_fix_prior != 'none':
        ax[0].plot(xrange, np.mean(all_results["fix_prior"], axis=-1), label = r'VAE $\kappa$-prior (fixed)', linewidth=1.8, linestyle=(0,(1,1)), color='#fbafe4')
    if args_new.config_file_path_pred_prior != 'none':
        ax[0].plot(xrange, np.mean(all_results["pred_prior"], axis=-1), label = r'VAE $\kappa$-prior (learned)', linewidth=1.8, linestyle=(0,(1,1)), color='#949494')
    if args_new.config_file_path_constr_only != 'none':
        ax[0].plot(xrange, np.mean(all_results["constr_only"], axis=-1), label = 'VAE constraints only', linewidth=1.8, linestyle=(0,(1,1)), color='#56b4e9')
    if args_new.flow_sample_path != 'none':
        ax[0].plot(xrange, np.mean(all_results["flow"], axis=-1), label = 'Flow', linewidth=1.8, linestyle='-.', color='#ece133')
    ax[0].set_ylabel('Variance (superposed)', fontsize=18, labelpad=10)
    ax[0].set_xlabel('Atom position', fontsize=18, labelpad=10)
    ax[0].set_ylim(0, 1.2 * MD.mean(axis=-1).max())
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    if not args_new.legend_off:
        ax[0].legend(fontsize=18, loc="upper center")

    ax[1].plot(xrange, np.mean(MD, axis=-1), label = 'MD reference', linewidth=3, color='tab:blue')
    ax[1].plot(xrange, np.mean(all_results["main"], axis=-1), label = 'VAE', linewidth=3, color='tab:orange')
    ax[1].plot(xrange, np.mean(npcov, axis=-1), label = 'Standard estimator', linewidth=1.8, linestyle='--', color='#029e73')
    if args_new.config_file_path_fix_prior != 'none':
        ax[1].plot(xrange, np.mean(all_results["fix_prior"], axis=-1), label = r'VAE $\kappa$-prior (fixed)', linewidth=1.8, linestyle=(0,(1,1)), color='#fbafe4')
    if args_new.config_file_path_pred_prior != 'none':
        ax[1].plot(xrange, np.mean(all_results["pred_prior"], axis=-1), label = r'VAE $\kappa$-prior (learned)', linewidth=1.8, linestyle=(0,(1,1)), color='#949494')
    if args_new.config_file_path_constr_only != 'none':
        ax[1].plot(xrange, np.mean(all_results["constr_only"], axis=-1), label = 'VAE constraints only', linewidth=1.8, linestyle=(0,(1,1)), color='#56b4e9')
    if args_new.flow_sample_path != 'none':
        ax[1].plot(xrange, np.mean(all_results["flow"], axis=-1), label = 'Flow', linewidth=1.8, linestyle='-.', color='#ece133')
    ax[1].set_ylabel('Variance (superposed)', fontsize=18, labelpad=10)
    ax[1].set_xlabel('Atom position', fontsize=18, labelpad=10)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    if not args_new.legend_off:
        ax[1].legend(fontsize=18, loc="upper center")

    fig.tight_layout()

    wandb.log({'atomfluct_plot_baselines':wandb.Image(fig)})

    save_fig = os.path.join(save_folder, f"{protein}_a{param_a}_mae{param_aux}_fluct_baselines.png")
    plt.savefig(save_fig, format='png')