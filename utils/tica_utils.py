import torch
import numpy as np
import mdtraj as md
import yaml
import os

class Args():
    """
    Class that serves as a replacement for argparser
    """
    def __init__(self, config_file_path, path_to_main):

        with open(config_file_path, "r") as f:
            config=yaml.safe_load(f)
        for (key, value) in config.items():
            if "path" in key:
                if value['value'].startswith('./'):
                    value['value'] = value['value'][2:]
                setattr(self, key, os.path.join(path_to_main, value['value']))
            elif not "wandb" in key and key not in ['batch_size', 'num_samples_z_eval']:
                setattr(self, key, value['value'])


def get_tic_features(xyz, top, bs):
    """
    Calculate features for TIC analysis, dihedrals and pairwise distances.
    """

    top_order_reverse = np.argsort(top["order"])
    top_order = np.argsort(top_order_reverse)
    if top["order"] is not None:
        xyz = xyz[:, top_order, :]

    num_batches = -(len(xyz) // -bs)

    for b in range(num_batches):
        xyz_batch = xyz[b*bs:(b+1)*bs]

        traj = md.Trajectory(xyz_batch.numpy() / 10, topology=top["topology"])

        ind = np.arange(0, xyz_batch.shape[1] - 3)
        ind = np.stack((ind, ind + 1, ind + 2, ind + 3)).T
        dihedrals = md.compute_dihedrals(traj, ind)

        if top_order_reverse is not None:
            xyz_batch = xyz_batch[:, top_order_reverse, :]

        pwds = get_pwd_triu_batch(xyz_batch).numpy()

        if b == 0:
            features = np.hstack((dihedrals, pwds))
        else:
            features = np.vstack((features, np.hstack((dihedrals, pwds))))

    return features


def get_pwd_triu_batch(x, offset=1):
    """
    Get pairwise distances (PWD) for a batch of structures,
    only for the upper triangle without the diagonal since
    the PWD matrix is symmetric and the diagonal is zero.
    So for structures with dimensions bs x num_beads x 3, this
    function returns bs x (num_beads**2-num_beads)/2 distances
    if offset=1 (default). Offset can also be specified to only
    take into account further off-diagonal distances.
    """
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Shape mismatch"
    pwd = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
    assert pwd.shape[-2] == pwd.shape[-1], "PWD matrix must be square"
    triu_ind = torch.triu_indices(pwd.shape[-2], pwd.shape[-1], offset=offset)
    return pwd[:, triu_ind[0], triu_ind[1]]