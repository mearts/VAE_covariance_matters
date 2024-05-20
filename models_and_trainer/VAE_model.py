import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, kl_divergence
import numpy as np
import mdtraj as md
from models_and_trainer.unet_model import UNet
from datetime import datetime
import warnings
import os
from utils.utils import (
    prepare_for_pnerf,
    samples_to_structures,
    get_dihedral_derivatives,
    get_bond_angle_derivatives,
    get_coord_with_O,
)


class VAE(nn.Module):
    def __init__(self, args, bond_lengths, prior):
        """
        Initialization of the model class
        """
        super(VAE, self).__init__()
        self.latent_features = args.latent_features
        self.encoder_sizes = args.encoder_list
        self.decoder_sizes = args.decoder_list
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eps = torch.finfo(torch.float).eps
        self.length = int(len(bond_lengths) * 3)
        self.upper_ind = torch.triu_indices(self.length, self.length, offset=1)
        self.num_dihedrals = self.length - 3
        self.num_bond_angles = self.length - 2
        self.num_kappa = 2 * self.length - 5
        self.bond_lengths = bond_lengths.to(self.device)
        self.mean_only = True

        self.constraints_off = args.constraints_off
        self.predict_prior =  args.predict_prior
        
        # Activations
        self.softplus = nn.Softplus()
        
        # Encoder
        in_features = self.num_kappa*2
        self.encoder = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=self.encoder_sizes[0]),
                    nn.BatchNorm1d(self.encoder_sizes[0]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.encoder_sizes[0], out_features=self.encoder_sizes[1]),
                    nn.BatchNorm1d(self.encoder_sizes[1]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.encoder_sizes[1], out_features=self.encoder_sizes[2]),
                    nn.BatchNorm1d(self.encoder_sizes[2]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.encoder_sizes[2], out_features=self.latent_features*2)
                    )
        
        # Decoder
        out_features = 2*self.num_kappa
        if self.predict_prior:
            out_features += self.num_kappa

        self.decoder = nn.Sequential(
                    nn.Linear(in_features=self.latent_features, out_features=self.decoder_sizes[0]),
                    nn.BatchNorm1d(self.decoder_sizes[0]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.decoder_sizes[0], out_features=self.decoder_sizes[1]),
                    nn.BatchNorm1d(self.decoder_sizes[1]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.decoder_sizes[1], out_features=self.decoder_sizes[2]),
                    nn.BatchNorm1d(self.decoder_sizes[2]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=self.decoder_sizes[2], out_features=out_features),
                    )
        self.tanh = nn.Tanh()
        
        # U-Net
        if not self.constraints_off:
            self.unet = UNet(n_channels=1, n_classes=1, extra_step=False, allow_negative=self.allow_negative_lambda)

        # Data prior prediction (if applicable)
        if self.predict_prior:
            warnings.warn("Predicting prior over kappa! Arguments prior and a_weight will be ignored.")
            self.prior = None
            a_weight = 1.
        else:
            if args.constraints_only:
                self.prior = torch.zeros_like(prior).to(self.device)
            else:
                self.prior = prior.to(self.device)
            a_weight = args.a_weight
            
        # Data prior weight
        self.a = torch.tensor([float(a_weight)], device = self.device)
        
        # Weight for auxiliary loss
        self.lamb_aux_weight = torch.tensor([float(args.lambda_aux_weight)], device = self.device)

    
    def get_kl_loss(self, mu_z, var_z):
        """
        Calculate KL divergence
        """

        prior = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        approx_post = Normal(mu_z, torch.sqrt(var_z+self.eps))
        
        KL_z = kl_divergence(approx_post, prior).sum(dim=-1)

        return KL_z.mean()


    def get_lambda_unet(self, structs):
        """
        Get lambda (lagrange multiplier) using a U-Net
        """
        if self.constraints_off:
            sf_average_pool_diag = torch.zeros((structs.shape[0], structs.shape[1]), device=self.device)
        else:
            pwds = torch.norm(structs[:, None, :, :]-structs[:, :, None, :], dim=-1)
            scale_factor_unet = self.unet(pwds.unsqueeze(dim=1)).squeeze(dim=1)

            # Average pool diagonal approach
            sf_rows = scale_factor_unet.sum(dim=-2)
            sf_cols = scale_factor_unet.sum(dim=-1)
            sf_diag = scale_factor_unet.diagonal(dim1=-2, dim2=-1)
            sf_average_pool_diag = (sf_rows + sf_cols - sf_diag) / (2 * scale_factor_unet.shape[1] - 1)

            assert sf_average_pool_diag.shape == (structs.shape[0], structs.shape[1]), 'Shape mismatch'
        return sf_average_pool_diag
        

    def get_prec_matrix(self, x, lamb, index=None, return_Cm=False):
        """ 
        Get precision matrix
        """
        if self.predict_prior:
            prior = torch.diag_embed(self.prior[index])
            assert prior.shape == (self.num_kappa, self.num_kappa), f"Shape mismatch, shape: {prior.shape}"
            prior = self.a * prior
        else:
            prior = self.a * self.prior

        if not self.constraints_off or return_Cm:
            dih_derivatives = get_dihedral_derivatives(x)
            ba_derivatives = get_bond_angle_derivatives(x)
            derivatives = torch.cat((dih_derivatives, ba_derivatives), dim=1)
            prec_constr = torch.sum(derivatives[:, :, None, :] * derivatives[:, None, :, :], dim = -1) # "G_m"
        if self.constraints_off:
            cov_new = prior
        else:
            cov_new = torch.einsum('m, mij->ij', (lamb, prec_constr))
            assert cov_new.shape == (self.num_kappa, self.num_kappa), 'Shape mismatch'
            cov_new += prior

        assert cov_new.isnan().sum() == 0, f"Precision contains {cov_new.isnan().sum()} nans"
        assert cov_new.isinf().sum() == 0, f"Precision contains {cov_new.isinf().sum()} infs"

        if return_Cm:
            cov_new = torch.linalg.inv(cov_new)

            Cm = cov_new.unsqueeze(dim=0) @ prec_constr
            Cm = Cm.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

            return cov_new, Cm
        else:
            return cov_new


    def get_structures(self, mu_k):
        """
        Get structures from mean over kappa
        """
        dih, ba = mu_k[:, :self.num_dihedrals], mu_k[:, self.num_dihedrals:]
        assert ba.shape[1] == self.num_bond_angles, 'Shape mismatch'

        di_pNeRF = prepare_for_pnerf(dih, kappa_type="di")
        ba_pNeRF = prepare_for_pnerf(ba, kappa_type="ba")
        structs = samples_to_structures(di_pNeRF, \
                      self.bond_lengths.repeat(1, len(mu_k), 1), ba_pNeRF)
                    
        return structs

    
    def get_NLL_AUX(self, mu_k, k_gt):
        """
        Get negative loglikelihood using full precision matrix
        Note: k_gt is already centered at zero!
        """
        structs = self.get_structures(mu_k)
        lamb = self.get_lambda_unet(structs)

        NLL = torch.tensor([0.], device=self.device)
        
        for i, (x, l) in enumerate(zip(structs, lamb)):
            prec = self.get_prec_matrix(x, l, i)
            x, l = None, None
            NLL += self.get_NLL_kappa(prec, k_gt, i)

        AUX = self.get_lambda_mae(lamb)

        return NLL / self.batch_size, AUX, {"mean":lamb.mean().item(), "min":lamb.min().item(), "max":lamb.max().item()}


    def get_NLL_kappa(self, prec, k_gt, i):
        """
        Get negative loglikelihood in kappa space
        """
        try:
            NLL = -MultivariateNormal(torch.zeros(self.num_kappa, device=self.device), precision_matrix=prec).log_prob(k_gt[i])
        except ValueError:
            if self.predict_prior:
                print(f"Prior: {self.prior[i]}")
            else:
                print(f"Prior: {self.prior}")
            print(f"Precision matrix diagonal: {prec.diag()}")
            print(f"Precision matrix: {prec}")
            raise
        
        return NLL


    def get_lambda_mae(self, lamb):
        """
        Auxiliary MAE loss on inverse lambdas to reduce fluctuations
        """
        AUX = (1/(lamb+self.eps)).mean(dim=1)

        return AUX.mean(dim=0)
    
    
    def get_recon_loss(self, mu_k, k_gt):
        """
        Get reconstruction loss. 
        """
        
        k_gt = torch.atan2(torch.sin(k_gt-mu_k), torch.cos(k_gt-mu_k))

        if self.mean_only:
            NLL = -Normal(torch.zeros_like(mu_k[0]), torch.ones_like(mu_k[0])).\
                  log_prob(k_gt).sum(dim=1).mean(dim=0)
            lamb = None
            AUX = torch.tensor([0.], device=self.device)
        else:
            NLL, AUX, lamb = self.get_NLL_AUX(mu_k, k_gt)
                   
        return NLL, AUX, lamb


    def sample_batches(self, num_samples_current, batch_size):
        """
        Get the number of batches for sampling
        """
        if batch_size is None:
            batches = 1
        elif num_samples_current % batch_size == 0:
            batches = num_samples_current // batch_size
        else:
            batches = num_samples_current // batch_size + 1
        return batches


    def lamb_in_batches(self, structs, batches, batch_size):
        """
        Get predicted lambda in batches
        """
        lamb = []
        for i in range(batches):
            end = structs.shape[0] if (i == batches-1) else (i+1)*batch_size
            lamb.append(self.get_lambda_unet(structs[i*batch_size:end]).cpu())
            assert lamb[i].shape[0] <= batch_size, 'Bug in batch size'
        lamb = torch.vstack(lamb)
        assert lamb.shape == (structs.shape[0], structs.shape[1]), f'Shape mismatch, shape: {lamb.shape}'

        return lamb


    def fluctuation_step(self, structs, lamb, mu_k):
        """
        Take one fluctation stepp (without calculating loss)
        """
        kappa = []
        prec_matrices = []
        for i, (x, l, k) in enumerate(zip(structs, lamb, mu_k)):
            prec = self.get_prec_matrix(x.to(self.device), l.to(self.device), i)
            MN = MultivariateNormal(torch.zeros(self.num_kappa, device=self.device), precision_matrix=prec)
            prec_matrices.append(prec.cpu())
            prec = None
            k = k.unsqueeze(0) + MN.sample(torch.Size([1]))
            kappa.append(torch.atan2(torch.sin(k), torch.cos(k)))

        return torch.vstack(kappa), torch.stack(prec_matrices)


    def sample(self, num_samples_z, topology = None, model_name = None, batch_size=None):
        """
        Sample from prior
        """
        self.eval()

        with torch.no_grad():
            z_prior = Normal(torch.zeros(self.latent_features, device = self.device), 
                            torch.ones(self.latent_features, device = self.device))
            z_samples = z_prior.sample(torch.Size([num_samples_z]))

            for i in range(2):
                if i == 0:
                    mu_k = self.decode(z_samples)
                elif i == 1:
                    batches = self.sample_batches(structs.shape[0], batch_size)
                    lamb = self.lamb_in_batches(structs, batches, batch_size)
                    structs = structs.cpu()

                    mu_k, prec = self.fluctuation_step(structs, lamb, mu_k)
                    prec = prec.cpu()
                    assert prec.shape == (num_samples_z, self.num_kappa, self.num_kappa), \
                                            f"Shape mismatch, shape: {prec.shape} instead of \
                                            {(num_samples_z, self.num_kappa, self.num_kappa)}"
                
                structs = self.get_structures(mu_k)
                assert structs.shape[0] == num_samples_z, \
                f'Shape mismatch, step {i}, shape: {structs.shape} (expected len {num_samples_z})'

                # Save sampled structures
                if topology is not None and i == 1:
                    if not os.path.isdir("./sample_pdb_files"):
                        os.makedirs("./sample_pdb_files")
                    traj = md.Trajectory(get_coord_with_O(structs, topology)/10, topology=topology["topology"])
                    model_name_save = "" if model_name is None else model_name + "_"
                    traj.save_pdb(f"pdb_files/samples_withO_step{i}of{self.steps}_{model_name_save}{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.pdb")

        return {'dihedrals':mu_k[:, :self.num_dihedrals].cpu(), 'bond_angles':mu_k[:, self.num_dihedrals:].cpu(), 
                'structures':structs.cpu(), 'precision_matrices':prec, 'z_samples':z_samples.cpu()}


    def sample_Cm(self, num_samples):
        """
        Get constraints Cm
        """
        self.eval()

        with torch.no_grad():
            z_prior = Normal(torch.zeros(self.latent_features, device = self.device), 
                            torch.ones(self.latent_features, device = self.device))
            z_samples = z_prior.sample(torch.Size([num_samples]))

            for i in range(2):
                if i == 0:
                    mu_k = self.decode(z_samples)
                elif i == 1:
                    lamb = self.get_lambda_unet(structs).cpu()
                    assert lamb.shape == (num_samples, structs.shape[1]), f'Shape mismatch, shape: {lamb.shape}'
                    structs = structs.cpu()

                    all_Cm = torch.empty_like(lamb)
                    for j, (x, l) in enumerate(zip(structs, lamb)):
                        _, all_Cm[j] = self.get_prec_matrix(x.to(self.device), l.to(self.device), j, return_Cm=True)
                
                structs = self.get_structures(mu_k)
                assert structs.shape[0] == num_samples, 'Shape mismatch'

        return all_Cm, lamb

    
    def encode(self, k):
        """
        Encode input
        """
        kshape = k.shape
        k = torch.flatten(torch.stack((torch.cos(k), torch.sin(k)), dim=2), start_dim=1, end_dim=2) # interleave cos and sin
        assert k.shape == (kshape[0], kshape[1]*2), f"Shape mismatch: k.shape = {k.shape} instead of {(kshape[0], kshape[1]*2)}"
        mu_z, var_z = torch.chunk(self.encoder(k), 2, dim=-1)
        var_z = self.softplus(var_z)
        z = Normal(mu_z, torch.sqrt(var_z+self.eps)).rsample()
        
        return mu_z, var_z, z


    def decode(self, z):
        """
        Decode from latent space
        """
        mu_k = self.decoder(z)

        if self.predict_prior:
            mu_k, var_k = torch.split(mu_k, (2*self.num_kappa, self.num_kappa), dim=-1)
            if not self.mean_only:
                var_k = self.softplus(var_k)
                self.prior = 1 / (var_k + self.eps)

        mu_k = self.tanh(mu_k) # Between -1 and 1
        mu_k = torch.atan2(mu_k[:, 1::2], mu_k[:, 0::2]) # [-pi, pi]
        assert (mu_k.min() >= -np.pi) and (mu_k.max() <= np.pi), "invalid dihedral (outside [-pi, pi])"
        
        return mu_k

    
    def forward(self, k, c, only_outputs=False):
        """
        Forward step of the model
        """
        k_in = k.clone()
        self.batch_size = len(k_in)

        mu_z, var_z, z = self.encode(k_in)

        mu_k = self.decode(z)
        z = z.cpu()
        
        if only_outputs:
            NLL, KL, AUX, lamb = None, None, None, None
        else:
            KL = self.get_kl_loss(mu_z, var_z)
            NLL, AUX, lamb = self.get_recon_loss(mu_k, k)

        return {"z":z, "kappa":mu_k}, NLL, KL, AUX, lamb
    
    

        