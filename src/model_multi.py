import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb

from src import utils
from src.egnn import DynamicsWithPockets
from src.edm_multi import EDM
from src.datasets import (
    create_templates_for_rgroup_generation_multi, get_dataloader, 
    MultiRDataset_anchor, collate_mr
)
from src.molecule_builder import build_molecules
from src.visualizer import visualize_chain, save_xyz_file_fa
from typing import Dict, List, Optional

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")


class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, include_charges, model,
        data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='scaffold', inpainting=False, anchors_context=True,
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        self.is_geom = True

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics = DynamicsWithPockets(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )
        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )

    def setup(self, stage: Optional[str] = None):
        dataset_type = MultiRDataset_anchor # anchors

        if stage == 'fit':
            self.is_geom = True
            self.train_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.is_geom = True
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def training_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=True)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)
        return training_metrics

    def validation_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def test_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

    def test_epoch_end(self, test_step_outputs):
        for metric in test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

    def generate_animation(self, chain_batch, node_mask, batch_i):
        batch_indices, mol_indices = utils.get_batch_idx_for_animation(self.batch_size, batch_i)
        for bi, mi in zip(batch_indices, mol_indices):
            chain = chain_batch[:, bi, :, :]
            name = f'mol_{mi}'
            chain_output = os.path.join(self.samples_dir, f'epoch_{self.current_epoch}', name)
            os.makedirs(chain_output, exist_ok=True)

            one_hot = chain[:, :, 3:-1] if self.include_charges else chain[:, :, 3:]
            positions = chain[:, :, :3]
            chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(self.FRAMES)], dim=0)
            names = [f'{name}_{j}' for j in range(self.FRAMES)]

            save_xyz_file_fa(chain_output, one_hot, positions, chain_node_mask, names=names)
            # save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names)

    # Using multiple anchors on the same graph as context to diff each rgroup
    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            rgroup_sizes = data['rgroup_mask'].sum(1).view(-1).int()
        else:
            rgroup_sizes = sample_fn(data)

        if self.inpainting:
            template_data = data
        else:
            template_data = create_templates_for_rgroup_generation_multi(data, rgroup_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        anchors = template_data['anchors']
        scaffold_mask = template_data['scaffold_mask']
        rgroup_mask = template_data['rgroup_mask']

        anchors_ = template_data['anchors_']
        scaffold_mask_batch_new = template_data['scaffold_mask_batch_new']
        scaffold_only_mask_batch_new = template_data['scaffold_only_mask_batch_new']
        x_batch_new = template_data['x_batch_new']
        h_batch_new = template_data['h_batch_new']
        node_mask_batch_new = template_data['node_mask_batch_new']
        scaffold_mask_ori_batch_new = template_data['scaffold_mask_ori_batch_new']
        rgroup_mask_batch_new = data['rgroup_mask_batch_new']
        rgroup_mask_ori_batch_new = template_data['rgroup_mask_ori_batch_new']
        batch_new_len_tensor = template_data['batch_new_len_tensor']

        # Anchors and scaffold labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors_, scaffold_mask_batch_new], dim=-1)
        else:
            context = scaffold_mask_batch_new

        # Add information about pocket to the context
        if '.' in self.train_data_prefix:
            scaffold_pocket_mask = scaffold_mask_batch_new
            scaffold_only_mask = scaffold_only_mask_batch_new
            pocket_only_mask = scaffold_pocket_mask - scaffold_only_mask
            if self.anchors_context:
                context = torch.cat([anchors_, scaffold_only_mask, pocket_only_mask], dim=-1)
            else:
                context = torch.cat([scaffold_only_mask, pocket_only_mask], dim=-1)

        # Removing COM
        if self.center_of_mass == 'scaffold':
            center_of_mass_mask = template_data['scaffold_only_mask']
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors_
        else:
            raise NotImplementedError(self.center_of_mass)
        
        x_masked = x_batch_new * center_of_mass_mask
        N = center_of_mass_mask.sum(1, keepdims=True)
        mean = torch.sum(x_masked, dim=1, keepdim=True) / N

        x = utils.remove_partial_mean_with_mask(x_batch_new, node_mask_batch_new, center_of_mass_mask)

        chain = self.edm.sample_chain(
            x=x,
            h=h_batch_new,
            node_mask=node_mask_batch_new,
            edge_mask=edge_mask,
            scaffold_mask=scaffold_mask_batch_new,
            rgroup_mask=rgroup_mask_batch_new,
            context=context,
            keep_frames=keep_frames,
        )

        return chain, node_mask_batch_new, mean

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()