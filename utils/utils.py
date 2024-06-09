import torch
from utils.pnerf_specify import dihedral_to_point, point_to_coordinate
from torch.utils.data import random_split
from scipy.stats import circvar
import numpy as np
import pandas as pd
import wandb
import os
import mdtraj as md
import warnings
import re
import warnings


def process_args(args):
    """
    Process arguments based on settings
    """

    if args.constraints_off:
        warnings.warn("Constraints are turned off! No auxiliary loss on lambda.")
        args.lambda_aux_weight = 0.0
        if args.predict_prior:
            args.model_name = "baseline_noconstr_predprior_decoder_" + args.model_name
        else:
            args.model_name = "baseline_noconstr_fixedprior_" + args.model_name

    if args.constraints_only:
        if args.a_weight > 0:
            warnings.warn("Constraints only! Prior is not used.")
        args.a_weight = 0
        args.predict_prior = False

    return args
    

def initialize(args):
    """
    Initialization stuff before loading data
    """
    # Device
    device, use_gpu = get_device()

    if not hasattr(args, "seed_off"):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        if not args.seed_off:
            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project,  entity=args.wandb_user)
        print(f"Wandb run dir: {wandb.run.dir}")
    wandb.config.update(vars(args), allow_val_change=True)

    save_folder = args.save_folder
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    return device, use_gpu, save_folder


def load_data(args):
    """
    Load data from different modalities
    """
    # In-house simulation data
    if args.protein == '1pga':
        coords = torch.from_numpy(np.load(args.data_file_path)).float()

        top = md.load(args.pdb_file_path).remove_solvent().topology
        ind_withO = None

    # NMR data from the Protein Data Bank
    elif args.protein in ['1unc', '1fsd']:
        warnings.warn('NMR data: data_file_path not used (only using pdb_file_path)')
        nmr = md.load(args.pdb_file_path).remove_solvent()
        ind = [atom.index for atom in nmr.topology.atoms if ((atom.name == 'CA') or \
            (atom.name == 'N') or (atom.name == 'C'))]
        coords = torch.from_numpy(nmr.atom_slice(ind).xyz * 10).float()

        top = nmr.topology
        ind_withO = None

    # D.E. Shaw Research data
    elif args.protein in ['chig', '2f4k']:
        path_list = os.listdir(args.data_file_path)
        dcd_files = list(filter(re.compile('.*.dcd').match, path_list))
        dcd_files = [os.path.join(args.data_file_path, f) for f in dcd_files]

        # Processing in batches if we have more than 1GB of data
        total_file_size = sum([os.path.getsize(f) for f in dcd_files]) * 1e-9
        batches = -(total_file_size // -1) # Ceiling division

        coords = []
        for i, dcd_batch in enumerate(np.array_split(dcd_files, batches)):
            traj = md.load(dcd_batch.tolist(), top=args.pdb_file_path)

            # Make sure atom ordering is correct
            if i == 0:
                unique_residues = pd.unique(np.array([str(r) for r in traj.topology.residues]))
                GT = np.array([[f"{r}-N",f"{r}-CA",f"{r}-C"] for r in unique_residues]).flatten().astype(str)
                naive = np.array([[atom, atom.index] for atom in traj.topology.atoms if ((atom.name == 'CA') or \
                        (atom.name == 'N') or (atom.name == 'C'))]).astype(str)
                ind = naive[:, 1][np.hstack([np.where(naive[:,0]==gt)[0] for gt in GT])].astype(int)
            coords_batch = torch.from_numpy(traj.atom_slice(ind).xyz * 10).float()

            coords.append(coords_batch)
        coords = torch.vstack(coords)   

        top = traj.topology
        # Indices for saving to pdb files later
        GT = np.array([[f"{r}-N",f"{r}-CA",f"{r}-C",f"{r}-O"] for r in unique_residues]).flatten()[:-1].astype(str)
        naive = np.array([atom for atom in traj.topology.atoms if ((atom.name == 'CA') or \
                (atom.name == 'N') or (atom.name == 'C') or (atom.name == 'O'))]).astype(str)
        naive = np.delete(naive, np.where(naive == f"{traj.topology.residue(-1)}-O")[0])
        ind_withO = np.hstack([np.where(naive==gt)[0] for gt in GT]).argsort()

    else:
        raise NotImplementedError(f"Data processing not implemented for given protein name: {args.protein}")

    top = top.subset(top.select('name CA or name N or name C or name O'))
    top.delete_atom_by_index(top.select(f'name O and resid {top.n_residues-1}')[0]) # Not placing last oxygen

    top = {"topology":top, "order":ind_withO}

    return coords, top


def process_data(args):
    """
    Prepare data for training or evaluation
    """
    coords, top = load_data(args)

    top_ref_ind = top["topology"].select('name CA or name N or name C')
    top_ref = top["topology"].subset(top_ref_ind)

    if top["order"] is not None:
        top_ref_order = top["order"][top_ref_ind]
        top_ref_order_reverse = np.argsort(top_ref_order)
        top_ref_order = np.argsort(top_ref_order_reverse)
    else:
        top_ref_order_reverse, top_ref_order = None, None

    if top_ref_order is not None:
        coords = coords[:, top_ref_order, :]
    coords = md.Trajectory(coords.numpy()/10, topology=top_ref)
    coords = coords.superpose(coords, frame=0).xyz*10
    if top_ref_order_reverse is not None:
        coords = coords[:, top_ref_order_reverse, :]
    coords = torch.from_numpy(coords).float()

    dihedrals = calculate_dihedrals(coords)
    bond_angles = calculate_bond_angles(coords)
    kappa = torch.cat((dihedrals.clone(), bond_angles.clone()), dim=1)
    kappa_clone = kappa.clone()
    bond_lengths = calculate_bond_lengths(coords)

    di, ba, bl = prepare_for_pnerf(dihedrals, kappa_type="di"), \
                prepare_for_pnerf(bond_angles, kappa_type="ba"), \
                prepare_for_pnerf(bond_lengths, kappa_type="bl")
    coords_pNeRF = samples_to_structures(di, bl, ba, use_gpu=False)

    # Split kappa and bond lengths
    num_val = int(0.1*len(kappa))
    ind_train, ind_val = random_split(torch.arange(len(kappa)), [len(kappa)-num_val, num_val])
    kappa_train, kappa_val = kappa[ind_train,:], kappa[ind_val,:]
    bond_lengths = bond_lengths[ind_train,:]

    print(f"Number of data points: {len(ind_train)} train and {len(ind_val)} validation.")

    coords_train, coords_val = coords[ind_train,:,:], coords[ind_val,:,:]

    # Prior (precision) for kappa fluctuations
    kappa_prior = torch.from_numpy(circvar(kappa_train, axis=0, low=-np.pi, high=np.pi))
    kappa_prior = torch.diag_embed(1/kappa_prior)

    # Median of bond lengths, for calculating structures with pNeRF
    bond_lengths = bond_lengths.median(dim=0, keepdim=True)[0] # TODO: median? Don't think it matters much...
    bond_lengths_pNeRF = prepare_for_pnerf(bond_lengths, kappa_type="bl")

    return {'kappa':kappa_clone, 'kappa_train':kappa_train, 'kappa_val':kappa_val, \
            'coords':coords, 'coords_train':coords_train, 'coords_val':coords_val, 'coords_pNeRF':coords_pNeRF, \
            'bond_lengths_pNeRF':bond_lengths_pNeRF, 'kappa_prior':kappa_prior, 'top':top, 'ind_train':ind_train}


def calculate_dihedrals(coords):
    """
    Calculate dihedrals, batch version
    """
    U = coords[:, 1:] - coords[:, :-1]        
    cross_12 = torch.cross(U[:, :-2], U[:, 1:-1])
    cross_23 = torch.cross(U[:, 1:-1], U[:, 2:])
    dihedrals = torch.atan2(torch.norm(U[:, 1:-1], dim=-1) * torch.sum(U[:, :-2] * cross_23, dim = -1), \
                            torch.sum(cross_12 * cross_23, dim = -1))

    return dihedrals


def calculate_bond_angles(coords):
    """
    Calculate bond angles, batch version
    """
    all_vectors = (coords[:, 1:] - coords[:, :-1])
    all_vectors = all_vectors / torch.norm(all_vectors, dim = -1, keepdim = True)

    return torch.acos(torch.einsum('kij, kij -> ki', all_vectors[:, 1:], -all_vectors[:, :-1]))

    
def calculate_bond_lengths(coords):
    """
    Calculate bond lengths, batch version
    """
    return torch.norm(100 * (coords[:, 1:] - coords[:, :-1]), dim = -1)


def prepare_for_pnerf(kappa, kappa_type):
    """
    Insert zeros for pNeRF
    kappa_type should be one of ["di","ba","bl"]
    """
    assert kappa_type in ["di","ba","bl"], "Invalid type of internal coordinate"
    zeros = torch.zeros((len(kappa),1), device=kappa.device)
    if kappa_type=="di":
        kappa = torch.cat((zeros, zeros, kappa, zeros), dim=-1)
    elif kappa_type=="ba":
        kappa = torch.cat((zeros, kappa, zeros), dim=-1)
    elif kappa_type=="bl":
        kappa = torch.cat((kappa, zeros), dim=-1)
    
    return kappa.reshape(kappa.shape[0], kappa.shape[1]//3, 3).transpose(0,1)


def samples_to_structures(dihedrals, bond_lengths, bond_angles, use_gpu=None):
    """ 
    Convert sampled dihedrals to structures using pNeRF
    :param dihedrals: Sampled dihedral sequences, shape = (num_samples, sequence_length, 3)
    """
    if use_gpu is None:
        use_gpu = True if torch.cuda.is_available() else False

    samples = dihedral_to_point(dihedrals, use_gpu, bond_lengths, bond_angles)
    samples = point_to_coordinate(samples, use_gpu = use_gpu).transpose(0, 1) / 100 # pm to angstrom
    return samples


def selected_inverse(matrix, type='inv'):
    if type == 'inv':
        return torch.linalg.inv(matrix)
    elif type == 'cholesky':
        return torch.cholesky_inverse(matrix)
    elif type == 'pseudo':
        return torch.linalg.pinv(matrix)
    else:
        raise NotImplementedError


def get_dihedral_derivatives(x, eps=1e-8):
    """
    Calculate atom fluctuations w.r.t. dihedral changes
    """
    pairs = x[None, :] - x[:, None]

    vector_chi = torch.diagonal(pairs, offset = 1).t()[1:-1]
    vector_chi = vector_chi / (torch.norm(vector_chi, dim = 1, keepdim = True)+eps) # Normalized vectors associated with dihedrals

    right_anchors = pairs.permute(2,0,1).triu(diagonal=1).permute(1,2,0)[2:-1].transpose(0,1)
    derivatives = torch.cross(*torch.broadcast_tensors(vector_chi[None, :, :], right_anchors))

    return derivatives


def get_bond_angle_derivatives(x, eps=1e-8):
    """
    Calculate atom fluctuations w.r.t. bond angle changes
    """
    pairs = x[None, :] - x[:, None]

    vector_ba = torch.diagonal(pairs, offset = 1).t()
    vector_ba = torch.cross(vector_ba[:-1], vector_ba[1:])
    vector_ba = vector_ba / (torch.norm(vector_ba, dim = 1, keepdim = True)+eps)

    right_anchors = pairs.permute(2,0,1).triu(diagonal=1).permute(1,2,0)[1:-1].transpose(0,1)
    derivatives = -torch.cross(*torch.broadcast_tensors(vector_ba[None, :, :], right_anchors))

    return derivatives


def get_device():
    """
    CPU or GPU?
    """
    if torch.cuda.is_available():
        device = 'cuda'
        use_gpu = True
    else:
        device = 'cpu'
        use_gpu = False

    print(f'Device: {device}')

    return device, use_gpu


def place_O(struct, sequence):
    """
    Place oxygen atom in backbone given N, Ca and C
    """
    
    all_pos_O = []
    
    bondLength_C_O = 1.231
    
    seq_Ca = struct[1::3][:-1]
    seq_C = struct[2::3][:-1]
    seq_N = struct[0::3][1:]
    
    for aa, ca, c, n in zip(sequence[:-1], seq_Ca, seq_C, seq_N):
        if aa == 'P':
            angle_O_C_N = 122.0 * np.pi/180
            bondLength_C_N = 1.341
        else:
            angle_O_C_N = 123.0 * np.pi/180
            bondLength_C_N = 1.329
        
        bc = (n-c) / bondLength_C_N
        n = np.cross((c-ca), bc)
        n = n / np.linalg.norm(n)
        nbc = np.cross(bc, n)   
        basis_change = np.stack((bc, nbc, n), axis=1)
        
        D = np.array([bondLength_C_O * np.cos(angle_O_C_N), bondLength_C_O * np.sin(angle_O_C_N), 0.0])
        
        pos_O = c + basis_change @ D
        all_pos_O.append(pos_O)
        
    return np.stack(all_pos_O)


def get_coord_with_O(coordinates, topology):
    """
    Place oxygens in N, Ca, C backbone and return new coordinate sequence
    """

    if coordinates.device.type != 'cpu':
        coordinates = coordinates.detach().cpu()

    new_coords = np.empty((len(coordinates), topology["topology"].n_atoms, 3))
    for i, c in enumerate(coordinates):
        pos_O = place_O(c.numpy(), topology["topology"].to_fasta()[0])
        pos_O = np.concatenate((pos_O, np.zeros((1,3))), axis=0)

        new_c = np.stack([c[0::3], c[1::3], c[2::3], pos_O], axis=0)
        new_coords[i] = new_c.transpose(1,0,2).reshape(-1, 3)[:-1]

    if topology["order"] is not None:
        new_coords = new_coords[:, topology["order"], :]

    return new_coords