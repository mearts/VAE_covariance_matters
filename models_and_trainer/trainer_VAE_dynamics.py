import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, MultivariateNormal, constraints
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet, empirical_covariance, oas
from scipy.stats import circmean
import wandb
import os
import mdtraj as md
from datetime import datetime
from utils.utils import prepare_for_pnerf, samples_to_structures, get_coord_with_O
import matplotlib
if int(matplotlib.__version__.split('.')[1]) >= 2:
    from matplotlib.colors import TwoSlopeNorm
else:
    from matplotlib.colors import DivergingNorm as TwoSlopeNorm
import warnings
    

class Trainer():
    def __init__(self, vae, args, data_dict, save_folder, start_epoch=0, eval_only=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vae = vae.to(self.device)
        self.opt = torch.optim.Adam(vae.parameters(), lr = args.lr)
        self.num_epochs = args.epochs
        self.num_warm_up = args.num_warm_up_KL
        self.num_mean_only = args.num_mean_only
        self.save_loc = os.path.join(save_folder, args.model_name + ".pt")
        self.start_epoch = start_epoch
        self.models_saved = 0

        # Data
        if eval_only:
            self.data, self.data_val = None, None
        else:
            drop_last_train = True if len(data_dict['kappa_train']) % args.batch_size == 1 else False # Avoid crashing when leftover batch size is 1
            self.data = TensorDataset(data_dict['kappa_train'], data_dict['coords_train'])
            self.data = DataLoader(self.data, batch_size=args.batch_size, shuffle=True, drop_last=drop_last_train)

            drop_last_val = True if len(data_dict['kappa_val']) % args.batch_size == 1 else False # Avoid crashing when leftover batch size is 1
            self.data_val = TensorDataset(data_dict['kappa_val'], data_dict['coords_val'])
            self.data_val = DataLoader(self.data_val, batch_size=args.batch_size, shuffle=False, drop_last=drop_last_val)

    def evaluate(self, warm_up_param, epoch=-1, val=False):
        self.vae.eval()
        with torch.no_grad():
            batch_losses_ELBO = 0
            batch_losses_NLL = 0
            batch_losses_KL = 0
            batch_losses_AUX = 0
            batch_losses_LOSS = 0

            data = self.data_val if val else self.data
            for i, (x, c) in enumerate(data):
                x = x.float().to(self.vae.device)
                _, NLL, KL, AUX, lamb = self.vae(x, c)
                ELBO = NLL + warm_up_param * KL
                LOSS = ELBO + self.vae.lamb_aux_weight * AUX
                batch_losses_ELBO += ELBO.item()
                batch_losses_NLL += NLL.item()
                batch_losses_KL += KL.item()
                batch_losses_AUX += AUX.item()
                batch_losses_LOSS += LOSS.item()

            if val:
                val_epoch = self.start_epoch if epoch == -1 else self.epoch
                wandb.log({'Epoch': val_epoch, 'Loss val': batch_losses_LOSS/(i+1), 'ELBO val': batch_losses_ELBO/(i+1), 
                        'NLL val': batch_losses_NLL/(i+1), 'KL val': batch_losses_KL/(i+1), 'AUX lamb val': batch_losses_AUX/(i+1)})
                if not self.vae.mean_only:
                    wandb.log({'Lambda val mean':lamb['mean'], 'Lambda val min':lamb['min'], 'Lambda val max':lamb['max']})
                print("Validation:\t%.4f" %(batch_losses_LOSS/(i+1)))
            else:
                wandb.log({'Epoch': self.start_epoch, 'Loss': batch_losses_LOSS/(i+1), 'ELBO': batch_losses_ELBO/(i+1), 
                        'NLL': batch_losses_NLL/(i+1), 'KL': batch_losses_KL/(i+1), 'AUX lamb': batch_losses_AUX/(i+1)})
                if not self.vae.mean_only:
                    wandb.log({'Lambda mean':lamb['mean'], 'Lambda min':lamb['min'], 'Lambda max':lamb['max']})
                    wandb.log({'AUX_lamb_weight':self.vae.lamb_aux_weight.item()})
                    wandb.log({'a':self.vae.a.item()})
                print("Batch average:\t%.4f" %(batch_losses_LOSS/(i+1)))

            if val and (epoch == self.num_mean_only or (self.epoch == self.start_epoch and self.start_epoch > 0)):
                self.best_loss = batch_losses_LOSS / (i+1)
            elif val and ((epoch > self.num_mean_only) and (batch_losses_LOSS / (i+1) < self.best_loss)):
                self.best_loss = batch_losses_LOSS / (i+1)

                # Only save if best loss and after warm-up
                if epoch > self.num_warm_up or self.start_epoch != 0:
                    self.models_saved += 1
                    self.save_model()
    
    def train(self):
        
        self.epoch = self.start_epoch
        if self.start_epoch==0:
            # Before training
            print("Before training...")
            warm_up_param = 0 if self.num_warm_up > 0 else 1
            self.vae.mean_only = True if self.num_mean_only > 0 else False
            self.evaluate(warm_up_param)
            self.evaluate(warm_up_param, val=True)
        else:
            # Checkpoint
            print("From checkpoint...")
            print(f"Epoch [{self.start_epoch}/{self.start_epoch + self.num_epochs}]")
            warm_up_param = self.start_epoch / self.num_warm_up if self.start_epoch < self.num_warm_up else 1.
            self.vae.mean_only = True if self.start_epoch < self.num_mean_only else False
            self.evaluate(warm_up_param)
            self.evaluate(warm_up_param, val=True)

        # Training
        self.vae.train()

        # Loop over epochs
        for epoch in range(self.num_epochs):
            self.epoch = self.start_epoch + epoch + 1
            print(f"Epoch [{self.epoch}/{self.start_epoch + self.num_epochs}]")
            warm_up_param = self.epoch / self.num_warm_up if self.epoch < self.num_warm_up else 1.
            self.vae.mean_only = True if self.epoch < self.num_mean_only else False

            batch_losses_LOSS = torch.empty(len(self.data))
            batch_losses_ELBO = torch.empty(len(self.data))
            batch_losses_NLL = torch.empty(len(self.data))
            batch_losses_KL = torch.empty(len(self.data))
            batch_losses_AUX = torch.empty(len(self.data))
            
            for i, (x, c) in enumerate(self.data):
                x = x.float().to(self.vae.device)
                c = c.float().to(self.vae.device) if 'x' in self.vae.ll else None
                self.opt.zero_grad()
                
                _, NLL, KL, AUX, lamb = self.vae(x, c)
                ELBO = NLL + warm_up_param * KL
                LOSS = ELBO + self.vae.lamb_aux_weight * AUX
                
                LOSS.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1)
                self.opt.step()

                batch_losses_LOSS[i] = LOSS.item()
                batch_losses_ELBO[i] = ELBO.item()
                batch_losses_NLL[i] = NLL.item()
                batch_losses_KL[i] = KL.item()
                batch_losses_AUX = AUX.item()

            wandb.log({'Epoch': self.epoch, 'Loss': torch.mean(batch_losses_LOSS), 'ELBO': torch.mean(batch_losses_ELBO), 
                       'NLL': torch.mean(batch_losses_NLL), 'KL':torch.mean(batch_losses_KL), 'AUX lamb':torch.mean(batch_losses_AUX)})
            if not self.vae.mean_only:
                wandb.log({'Lambda mean':lamb['mean'], 'Lambda min':lamb['min'], 'Lambda max':lamb['max']})
                wandb.log({'AUX_lamb_weight':self.vae.lamb_aux_weight.item()})
                wandb.log({'a':self.vae.a.item()})
            print("Batch average:\t%.4f" %(torch.mean(batch_losses_LOSS)))

            # Evaluate
            self.evaluate(warm_up_param, self.epoch, val=True)
            self.vae.train()

        if self.models_saved == 0:
            warnings.warn("No model with the best loss after warmup. Saving last model instead.")
            self.save_model()


    def save_model(self):
        """
        Save model and optimizer state dict
        """
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'length': self.vae.length,
                    'bond_lengths': self.vae.bond_lengths.detach().cpu(),
                    'prior': self.vae.prior.detach().cpu()}, self.save_loc)
        print(f"Saved model at {self.save_loc}")


    def get_basic_results(self, plot_data, num_samples_z, topology, model_name, batch_size):
        """
        Get results for general use in results plots
        """
        z = []
        for x in plot_data:
            output,_,_,_,_ = self.vae(x.to(self.device), None, only_outputs=True)
            z.append(output["z"])
        x = None
        plot_data = None
        z = torch.vstack(z)
        if z.shape[1] > 2: 
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)
            pca = None
        
        samples = self.vae.sample(num_samples_z, topology, model_name, batch_size=batch_size)

        return z, samples


    def plot_latent_and_matrices(self, z, samples, save):
        """
        Plot latent space and matrices.
        """
        di = samples['dihedrals'].cpu().numpy()
        ba = samples['bond_angles'].cpu().numpy()
        assert di.shape[-1] == self.vae.num_dihedrals, f'Shape mismatch, shape: {di.shape}'
        pwds = torch.norm(samples['structures'][:, None, :, :] - samples['structures'][:,:, None, :], dim=-1)
        d_mean, d_std = pwds.mean(dim=0).cpu().numpy(), pwds.std(dim=0).cpu().numpy()
        precs = samples["precision_matrices"].cpu().numpy()

        fig1, ax = plt.subplots(4, 2, figsize=(12, 24), facecolor = 'white')
        ax = ax.flatten()

        if precs.mean(axis=0).min().item() >= 0:
            norm = TwoSlopeNorm(vmin=-0.0001, vmax = precs.mean(axis=0).max().item(), vcenter=0)
        elif precs.mean(axis=0).max().item() <= 0:
            norm = TwoSlopeNorm(vmin=precs.mean(axis=0).min().item(), vmax = 0.0001, vcenter=0)
        else:   
            norm = TwoSlopeNorm(vmin=precs.mean(axis=0).min().item(), vmax = precs.mean(axis=0).max().item(), vcenter=0)

        sns.heatmap(data=precs.mean(axis=0), cbar = True, square = True, annot = False, cmap = 'bwr', xticklabels = [], yticklabels = [], ax=ax[0], norm=norm)
        ax[0].set_title("Precision matrix (mean)", fontsize = 20, pad = 50)
        ax[0].set_xlim(0, precs.shape[1])
        ax[0].set_ylim(0, precs.shape[1])
        cbar = ax[0].collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        sns.heatmap(data=precs[0], cbar = True, square = True, annot = False, cmap = 'bwr', xticklabels = [], yticklabels = [], ax=ax[1], norm=norm)
        ax[1].set_title("Precision matrix [0]", fontsize = 20, pad = 50)
        ax[1].set_xlim(0, precs.shape[1])
        ax[1].set_ylim(0, precs.shape[1])
        cbar = ax[1].collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        sns.scatterplot(x=z[:, 0], y=z[:, 1], alpha=0.5, linewidth=0, color = 'green', ax=ax[2])
        ax[2].set_title("Z", fontsize = 20, y = 1.04)
        ax[2].set_xlabel("z1", fontsize = 18, labelpad = 10)
        ax[2].set_ylabel("z2", fontsize = 18, labelpad = 10)

        ax[3].hist(ba[:, 0::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_{Ca}$')
        ax[3].hist(ba[:, 1::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_C$')
        ax[3].hist(ba[:, 2::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\theta_N$')
        ax[3].set_title("\nBond angles", fontsize = 20, y = 1.04)
        ax[3].set_xlabel("Angle", fontsize=18, labelpad=10)
        ax[3].set_ylabel("Density", fontsize=18, labelpad=10)
        ax[3].tick_params(axis='both', which='major', labelsize=16)
        ax[3].legend(fontsize=18)

        ax[4].hist(di[:, 0::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\psi$')
        ax[4].hist(di[:, 1::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\omega$')
        ax[4].hist(di[:, 2::3][:, 1:-1].flatten(), bins = 40, alpha = 0.5, density = True, label=r'$\phi$')
        ax[4].set_title("\nDihedrals", fontsize = 20, y = 1.04)
        ax[4].set_xlabel("Angle", fontsize=18, labelpad=10)
        ax[4].set_ylabel("Density", fontsize=18, labelpad=10)
        ax[4].set_xlim(-np.pi-0.2, np.pi+0.2)
        ax[4].set_xticks([-3,-2,-1,0,1,2,3])
        ax[4].tick_params(axis='both', which='major', labelsize=16)
        ax[4].legend(fontsize=18)

        sns.scatterplot(x=(di[:, 2::3][:, 1:-1]).flatten(), y=(di[:, 0::3][:, 1:-1]).flatten(), alpha = 0.1, linewidth=0,  ax=ax[5])#, label="Samples")
        ax[5].set_title("\nRamachandran", fontsize = 20, y = 1.04)
        ax[5].set_xlabel(r"$\phi$", fontsize = 18, labelpad = 10)
        ax[5].set_ylabel(r"$\psi$", fontsize = 18, labelpad = 10)
        ax[5].set_xlim(-np.pi-0.2, np.pi+0.2)
        ax[5].set_xticks([-3,-2,-1,0,1,2,3])
        ax[5].set_yticks([-3,-2,-1,0,1,2,3])
        ax[5].set_ylim(-np.pi-0.2, np.pi+0.2)
        ax[5].tick_params(axis='both', which='major', labelsize=16)

        sns.heatmap(data=d_mean, cbar = True, square = True, annot = False, cmap = 'magma', xticklabels = [], yticklabels = [], ax=ax[6])
        ax[6].set_title("\nMean distances", fontsize = 20, pad = 30)
        ax[6].set_xlim(0, d_mean.shape[1])
        ax[6].set_ylim(0, d_mean.shape[1])

        sns.heatmap(data=d_std, cbar = True, square = True, annot = False, cmap = 'magma', xticklabels = [], yticklabels = [], ax=ax[7])
        ax[7].set_title("\n"+r"$\sigma$ distances", fontsize = 20, pad = 30)
        ax[7].set_xlim(0, d_mean.shape[1])
        ax[7].set_ylim(0, d_mean.shape[1])
        
        fig1.tight_layout()

        wandb.log({'sample_plot':wandb.Image(fig1)})

        if save is not None:
            plt.savefig(save, format='png')


    def plot_fluctuation(self, samples, kappa, topology, save, num_samples_z, coords, coords_pnerf, model_name, bs_Cm=None):
        """
        Fluctuation plots
        """

        top2_ind = topology["topology"].select('name CA or name N or name C')
        top2 = topology["topology"].subset(top2_ind)

        if topology["order"] is not None:
            top2_order = topology["order"][top2_ind]
            top2_order_reverse = np.argsort(top2_order)
            top2_order = np.argsort(top2_order_reverse)
        else:
            top2_order_reverse, top2_order = None, None

        # MD
        if top2_order is not None:
            coords = coords[:, top2_order, :]
        traj_MD = md.Trajectory(coords.numpy()/10, topology=top2)
        traj_MD = traj_MD.superpose(traj_MD, frame=0).xyz*10
        if top2_order_reverse is not None:
            traj_MD = traj_MD[:, top2_order_reverse, :]
        MD = np.var(traj_MD, axis=0)
        MD_ref = traj_MD[0]

        # Samples
        structs = samples['structures'].clone()
        xrange = np.arange(structs.shape[1])
        structs = np.concatenate((np.expand_dims(MD_ref, axis=0), structs.numpy()), axis=0)
        if top2_order is not None:
            structs = structs[:, top2_order, :]
        traj_samp = md.Trajectory(structs/10, topology=top2)
        samp_unaligned = traj_samp.xyz[1:]*10
        if top2_order_reverse is not None:
            samp_unaligned = samp_unaligned[:, top2_order_reverse, :]
        samp_unaligned = np.var(samp_unaligned, axis=0)

        samp_aligned = traj_samp.superpose(traj_samp, frame=0).xyz[1:]*10
        if top2_order_reverse is not None:
            samp_aligned = samp_aligned[:, top2_order_reverse, :]
        samp_aligned = np.var(samp_aligned, axis=0)

        # Cm
        num_samp_Cm = len(structs)-1
        if bs_Cm is not None:
            cm_batches = -(num_samp_Cm // -bs_Cm)
            cm_residual = num_samp_Cm % bs_Cm
            Cm = []
            lamb = []
            for b in range(cm_batches):
                nsamp = cm_residual if (b == (cm_batches-1) and cm_residual != 0) else bs_Cm
                Cm_batch, lamb_batch = self.vae.sample_Cm(num_samples=nsamp)
                Cm.append(Cm_batch.numpy())
                lamb.append(lamb_batch.numpy())
            Cm = np.vstack(Cm)
            lamb = np.vstack(lamb)
        else:
            Cm, lamb = self.vae.sample_Cm(num_samples=num_samp_Cm)
            Cm, lamb = Cm.numpy(), lamb.numpy()
        Cm_mean = np.mean(Cm, axis=0)
        Cm_std = np.std(Cm, axis=0)

        # Prior
        prior = self.vae.data_prior.to(self.device) if self.vae.predict_prior else self.vae.prior
        prior_diag = self.vae.a.item() * prior.diag()
        prior_distr = Normal(torch.zeros(len(prior_diag), device=self.device), torch.sqrt(1/(prior_diag)))
        prior_samples = prior_distr.sample(torch.Size([num_samples_z]))
        prior_samples += torch.from_numpy(circmean(kappa, axis=0, low=-np.pi, high=np.pi)).unsqueeze(dim=0).to(self.device)
        assert prior_samples.shape[1:] == kappa.shape[1:], f"Shape mismatch, kappa shape: {kappa.shape}, samples shape: {prior_samples.shape}"
        prior_samples = torch.atan2(torch.sin(prior_samples), torch.cos(prior_samples))
        dih_prior, ba_prior = prior_samples[:, :self.vae.num_dihedrals], prior_samples[:, self.vae.num_dihedrals:]

        dih_pNeRF = prepare_for_pnerf(dih_prior, kappa_type="di")
        ba_pNeRF = prepare_for_pnerf(ba_prior, kappa_type="ba")
        structs_prior = samples_to_structures(dih_pNeRF,self.vae.bond_lengths.repeat(1, num_samples_z, 1), ba_pNeRF).cpu()
        if top2_order is not None:
            structs_prior = structs_prior[:, top2_order, :]
        traj_prior = md.Trajectory(structs_prior.numpy(), topology=top2)
        traj_prior = traj_prior.superpose(traj_prior, frame=0).xyz
        if top2_order_reverse is not None:
            traj_prior = traj_prior[:, top2_order_reverse, :]
            structs_prior = structs_prior[:, top2_order_reverse, :]
        prior = np.var(traj_prior, axis=0)

        if not os.path.isdir("./pdb_files"):
            os.makedirs("./pdb_files")
        traj = md.Trajectory(get_coord_with_O(structs_prior, topology)/10, topology=topology["topology"])
        model_name = "" if model_name is None else model_name + "_"
        traj.save_pdb("pdb_files/prior_withO_" + model_name + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".pdb")

        # np.cov
        kappa_mean = torch.from_numpy(circmean(kappa, axis=0, low=-np.pi, high=np.pi))
        kappa_offsets = torch.atan2(torch.sin(kappa - kappa_mean), torch.cos(kappa - kappa_mean))

        try:
            kappa_npcov = torch.from_numpy(empirical_covariance(kappa_offsets)).float().to(self.device)
            MN = MultivariateNormal(torch.zeros_like(kappa[0], device=self.device), covariance_matrix=kappa_npcov)
            covlabel = 'empirical'
        except ValueError:
            try:
                kappa_npcov = torch.from_numpy(MinCovDet(random_state=42).fit(kappa_offsets).covariance_).float().to(self.device)
                MN = MultivariateNormal(torch.zeros_like(kappa[0], device=self.device), covariance_matrix=kappa_npcov)
                covlabel = 'robust, MinCovDet'
            except ValueError:
                kappa_npcov = torch.from_numpy(oas(kappa_offsets, assume_centered=True)[0]).float().to(self.device)
                MN = MultivariateNormal(torch.zeros_like(kappa[0], device=self.device), covariance_matrix=kappa_npcov)
                covlabel = 'shrunk, OAS'

        
        npcov_samples = MN.sample(torch.Size([num_samples_z])) + kappa_mean.to(self.device)
        npcov_samples = torch.atan2(torch.sin(npcov_samples), torch.cos(npcov_samples))
        dih_npcov, ba_npcov = npcov_samples[:, :self.vae.num_dihedrals], npcov_samples[:, self.vae.num_dihedrals:]

        di_pNeRF = prepare_for_pnerf(dih_npcov, kappa_type="di")
        ba_pNeRF = prepare_for_pnerf(ba_npcov, kappa_type="ba")
        structs_npcov = samples_to_structures(di_pNeRF, self.vae.bond_lengths.repeat(1, num_samples_z, 1), ba_pNeRF).cpu()
        if top2_order is not None:
            structs_npcov = structs_npcov[:, top2_order, :]
        traj_npcov = md.Trajectory(structs_npcov.numpy()/10, topology=top2)
        traj_npcov = traj_npcov.superpose(traj_npcov, frame=0).xyz*10
        if top2_order_reverse is not None:
            traj_npcov = traj_npcov[:, top2_order_reverse, :]
            structs_npcov = structs_npcov[:, top2_order_reverse, :]
        npcov = np.var(traj_npcov, axis=0)

        if not os.path.isdir("./pdb_files"):
            os.makedirs("./pdb_files")
        traj = md.Trajectory(get_coord_with_O(structs_npcov, topology)/10, topology=topology["topology"])
        model_name = "" if model_name is None else model_name
        traj.save_pdb("pdb_files/estcov_withO_" + model_name + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".pdb")

        fig2, ax = plt.subplots(7, 1, figsize=(10, 35), facecolor = 'white')
        ax = ax.flatten()

        ax[0].plot(xrange, np.mean(samp_unaligned, axis=-1), label = 'VAE', linewidth=3, color='tab:orange')
        ax[0].plot(xrange, Cm_mean, label = r"$C$", linestyle = '--', linewidth=2, color='tab:grey')
        ax[0].fill_between(xrange, Cm_mean-Cm_std, Cm_mean+Cm_std, color='tab:grey', alpha=0.1)
        ax[0].set_ylabel('Variance (non-superposed)', fontsize=18, labelpad=10)
        ax[0].set_xlabel('Atom position', fontsize=18, labelpad=10)
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].legend(fontsize=18)

        ax[1].plot(xrange, np.mean(np.var(coords_pnerf.numpy(), axis=0), axis=-1), label = 'MD', linewidth = 3, color='tab:blue')
        ax[1].plot(xrange, np.mean(samp_unaligned, axis=-1), label = 'Samples', linewidth=3, color='tab:orange')
        ax[1].set_ylabel('Variance (non-superposed)', fontsize=18, labelpad=10)
        ax[1].set_xlabel('Atom position', fontsize=18, labelpad=10)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].legend(fontsize=18, loc="upper center")

        ax[2].plot(xrange, np.mean(MD, axis=-1), label = 'MD reference', linewidth = 3, color='tab:blue')
        ax[2].plot(xrange, np.mean(samp_aligned, axis=-1), label = 'VAE', linewidth=3, color='tab:orange')
        ax[2].plot(xrange, np.mean(prior, axis=-1), label = 'Prior', linestyle='--', color='tab:green')
        ax[2].plot(xrange, np.mean(npcov, axis=-1), label = 'Standard estimator\n'+covlabel, linestyle='--', color='tab:red')
        ax[2].set_ylabel('Variance (superposed)', fontsize=18, labelpad=10)
        ax[2].set_xlabel('Atom position', fontsize=18, labelpad=10)
        ax[2].set_ylim(0, 1.2 * MD.mean(axis=-1).max())
        ax[2].tick_params(axis='both', which='major', labelsize=16)
        ax[2].legend(fontsize=18, loc="upper center")

        ax[3].plot(xrange, np.mean(MD, axis=-1), label = 'MD reference', linewidth = 3, color='tab:blue')
        ax[3].plot(xrange, np.mean(samp_aligned, axis=-1), label = 'VAE', linewidth=3, color='tab:orange')
        ax[3].plot(xrange, np.mean(prior, axis=-1), label = 'Prior', linestyle='--', color='tab:green')
        ax[3].plot(xrange, np.mean(npcov, axis=-1), label = f'Standard estimator\n({covlabel})', linestyle='--', color='tab:red')
        ax[3].set_ylabel('Variance (superposed)', fontsize=18, labelpad=10)
        ax[3].set_xlabel('Atom position', fontsize=18, labelpad=10)
        ax[3].tick_params(axis='both', which='major', labelsize=16)
        ax[3].legend(fontsize=18, loc="upper center")

        for l, c in zip(lamb, Cm):
            ax[4].scatter(l, c, alpha=0.4, s=14)
        ax[4].set_ylabel(r'$C_m$', fontsize=18, labelpad=10)
        ax[4].set_xlabel(r'$\lambda$', fontsize=18, labelpad=10)
        ax[4].tick_params(axis='both', which='major', labelsize=16)

        assert sum(~constraints.positive.check(Cm.flatten() + self.vae.eps))==0, f"Cm not positive, Cm_min:{Cm.flatten().min()}, Cm_max:{Cm.flatten().max()}, Cm:{Cm}"
        ax[5].set_yscale('log')
        for l, c in zip(lamb, Cm):
            ax[5].scatter(l, c + self.vae.eps, alpha=0.4, s=14)
        ax[5].set_ylabel(r'$C_m$ (log scale)', fontsize=18, labelpad=10)
        ax[5].set_xlabel(r'$\lambda$', fontsize=18, labelpad=10)
        ax[5].tick_params(axis='both', which='major', labelsize=16)

        assert sum(~constraints.positive.check(lamb.flatten() + self.vae.eps))==0, f"Lamb not >0, Lamb_min:{lamb.flatten().min()}, Lamb_max:{lamb.flatten().max()}, Lamb:{lamb}"
        ax[6].set_xscale('log')
        for l, c in zip(lamb, Cm):
            ax[6].scatter(l + self.vae.eps, c, alpha=0.4, s=14)
        ax[6].set_ylabel(r'$C_m$', fontsize=18, labelpad=10)
        ax[6].set_xlabel(r'$\lambda$ (log scale)', fontsize=18, labelpad=10)
        ax[6].tick_params(axis='both', which='major', labelsize=16)

        fig2.tight_layout()

        wandb.log({'atomfluct_plot':wandb.Image(fig2)})

        save_fig2 = "." + "".join(save.split('.')[:-1]) if save.startswith(".") else save.split('.')[0]
        save_fig2 += '_fluct.png'
        plt.savefig(save_fig2, format='png')
        
        
        
    def get_plots(self, plot_data, coords, coords_pnerf, topology, num_samples_z=100, batch_size=None, save=None, model_name=None):
        """ 
        Plot latent space and statistics for sampled structures
        """ 
        self.vae.eval()

        with torch.no_grad():
            print(f'a: {self.vae.a.item()}')
            if batch_size == None:
                batch_size = len(plot_data)

            kappa = plot_data.clone()
            di_means = circmean(plot_data[:, :self.vae.num_dihedrals].cpu().numpy(), axis=0, low=-np.pi, high=np.pi)
            plot_data = DataLoader(plot_data, batch_size=batch_size, shuffle=False)

            z, samples = self.get_basic_results(plot_data, num_samples_z, topology, model_name, batch_size)
            
            # Plot 1: latent space and matrices
            self.plot_latent_and_matrices(z, samples, di_means, save)

            # Plot 2: fluctuation plots
            self.plot_fluctuation(samples, kappa, topology, save, num_samples_z, coords, coords_pnerf, model_name, bs_Cm=batch_size)
        
            

