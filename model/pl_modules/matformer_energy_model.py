import time

import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import grad
from tqdm import tqdm

from model.features import angle_emb_mp
from model.model_utils import RBFExpansion
from model.transformer import MatformerConv
from tools.utils import PROJECT_ROOT
from tools.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, convert_AFL_to_graphs)
MAX_ATOMIC_NUM=100

from model.pl_modules.diff_utils import d_log_p_wrapped_normal
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
import pdb


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
def calc_grouped_angles_mean_in_radians(data_tensor, groups):
    # 将[0,1]映射为圆上的弧度
    data_tensor = data_tensor * 2*math.pi

    # 初始化一个 tensor 用于保存分组后的数据
    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    # 使用 scatter 计算每个组的累加和
    sum_sin = scatter(data_tensor_sin, groups, dim=0, reduce='sum')
    sum_cos = scatter(data_tensor_cos, groups, dim=0, reduce='sum')

    # 计算每个组的数量
    group_counts = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce='sum')

    # 计算分组的平均值
    mean_sin = sum_sin / group_counts
    mean_cos = sum_cos / group_counts

    mean_angle = torch.atan2(mean_sin, mean_cos)  # Calculate mean angle in radians using atan2 for stability

    # Adjust mean_angle to be in the range [0, 2*pi)
    # mean_angle = torch.where(mean_angle >= 0, mean_angle, mean_angle + 2 * math.pi)
    mean_angle = mean_angle % (2*math.pi)

    # 圆上的弧度映射回[0, 1]
    mean_angle = mean_angle / (2*math.pi)

    return mean_angle

### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError

class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)

class Matformer_pl(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # super().__init__()
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type
        self.decoder = hydra.utils.instantiate(self.hparams.decoder,
                                               latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
                                               pred_type=True, pred_scalar=True, smooth=True)

        # self.classification = self.hparams.classification
        # self.use_angle = self.hparams.use_angle
        # self.atom_embedding = nn.Linear(
        #     self.hparams.atom_input_features, self.hparams.node_features
        # )
        # self.rbf = nn.Sequential(
        #     RBFExpansion(
        #         vmin=0,
        #         vmax=8.0,
        #         bins=self.hparams.edge_features,
        #     ),
        #     nn.Linear(self.hparams.edge_features, self.hparams.node_features),
        #     nn.Softplus(),
        #     nn.Linear(self.hparams.node_features, self.hparams.node_features),
        # )
        # self.angle_lattice = self.hparams.angle_lattice
        # if self.angle_lattice:  ## module not used
        #     print('use angle lattice')
        #     self.lattice_rbf = nn.Sequential(
        #         RBFExpansion(
        #             vmin=0,
        #             vmax=8.0,
        #             bins=self.hparams.edge_features,
        #         ),
        #         nn.Linear(self.hparams.edge_features, self.hparams.node_features),
        #         nn.Softplus(),
        #         nn.Linear(self.hparams.node_features, self.hparams.node_features)
        #     )
        #
        #     self.lattice_angle = nn.Sequential(
        #         RBFExpansion(
        #             vmin=-1,
        #             vmax=1.0,
        #             bins=self.hparams.triplet_input_features,
        #         ),
        #         nn.Linear(self.hparams.triplet_input_features, self.hparams.node_features),
        #         nn.Softplus(),
        #         nn.Linear(self.hparams.node_features, self.hparams.node_features)
        #     )
        #
        #     self.lattice_emb = nn.Sequential(
        #         nn.Linear(self.hparams.node_features * 6, self.hparams.node_features),
        #         nn.Softplus(),
        #         nn.Linear(self.hparams.node_features, self.hparams.node_features)
        #     )
        #
        #     self.lattice_atom_emb = nn.Sequential(
        #         nn.Linear(self.hparams.node_features * 2, self.hparams.node_features),
        #         nn.Softplus(),
        #         nn.Linear(self.hparams.node_features, self.hparams.node_features)
        #     )
        #
        # self.edge_init = nn.Sequential(  ## module not used
        #     nn.Linear(3 * self.hparams.node_features, self.hparams.node_features),
        #     nn.Softplus(),
        #     nn.Linear(self.hparams.node_features, self.hparams.node_features)
        # )
        #
        # self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0)  ## module not used
        #
        # self.angle_init_layers = nn.Sequential(  ## module not used
        #     nn.Linear(120, self.hparams.node_features),
        #     nn.Softplus(),
        #     nn.Linear(self.hparams.node_features, self.hparams.node_features)
        # )
        #
        # self.att_layers = nn.ModuleList(
        #     [
        #         MatformerConv(in_channels=self.hparams.node_features, out_channels=self.hparams.node_features,
        #                       heads=self.hparams.node_layer_head, edge_dim=self.hparams.node_features)
        #         for _ in range(self.hparams.conv_layers)
        #     ]
        # )
        #
        # self.edge_update_layers = nn.ModuleList(  ## module not used
        #     [
        #         MatformerConv(in_channels=self.hparams.node_features, out_channels=self.hparams.node_features,
        #                       heads=self.hparams.edge_layer_head, edge_dim=self.hparams.node_features)
        #         for _ in range(self.hparams.edge_layers)
        #     ]
        # )
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hparams.node_features, self.hparams.fc_features), nn.SiLU()
        # )
        # self.sigmoid = nn.Sigmoid()
        #
        # if self.classification:
        #     self.fc_out = nn.Linear(self.hparams.fc_features, 2)
        #     self.softmax = nn.LogSoftmax(dim=1)
        # else:
        #     self.fc_out = nn.Linear(
        #         self.hparams.fc_features, self.hparams.output_features
        #     )
        #
        # self.link = None
        # self.link_name = self.hparams.link
        # if self.hparams.link == "identity":
        #     self.link = lambda x: x
        # elif self.hparams.link == "log":
        #     self.link = torch.exp
        #     avg_gap = 0.7  # magic number -- average bandgap in dft_3d
        #     if not self.zero_inflated:
        #         self.fc_out.bias.data = torch.tensor(
        #             np.log(avg_gap), dtype=torch.float
        #         )
        # elif self.hparams.link == "logit":
        #     self.link = torch.sigmoid

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        times0 = torch.zeros(batch_size).to(self.device)
        time0_emb = self.time_embedding(times0)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        num_atoms = len(frac_coords)

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(
            batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        if self.update_type:
            atom_type_probs_noised = (
                    c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(
                batch.num_atoms)[:, None] * rand_t)
        else:
            atom_type_probs_noised = gt_atom_types_onehot

        # concat t,[A~,A],[F~,F],[L~,L],batch.num_atoms,batch.batch
        conc_time_emb = torch.cat((time_emb, time0_emb), dim=0)
        conc_atom_type_probs = torch.cat((atom_type_probs_noised, gt_atom_types_onehot), dim=0)
        conc_frac_coords = torch.cat((input_frac_coords, frac_coords), dim=0)
        conc_input_lattice = torch.cat((input_lattice, lattices), dim=0)
        batch_num_atoms = torch.cat((batch.num_atoms, batch.num_atoms), dim=0)
        batch_batch = torch.cat((batch.batch, (batch.batch + batch_size)), dim=0)

        if self.hparams.denoising is True:
            graphs =  convert_AFL_to_graphs(input_lattice,frac_coords,batch)
            # pred_l, pred_x, pred_t, pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms,
            #                       batch.batch)

            pred_e, pred_force = self.decoder(graphs.to(self.device))
            tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
            # loss_lattice = F.mse_loss(pred_l, rand_l)
            # loss_coord = F.mse_loss(pred_x, tar_x)
            # loss_type = F.mse_loss(pred_t, rand_t)
        else:
            pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        loss_energy = F.l1_loss(pred_e, batch.y)

        loss = loss_energy

        # if self.hparams.denoising is True:
        #     loss = (loss_energy +
        #             self.hparams.cost_lattice * loss_lattice +
        #             self.hparams.cost_coord * loss_coord +
        #             self.hparams.cost_type * loss_type)

        return {
            'loss': loss,
            'loss_energy': loss_energy,
            # 'loss_lattice': loss_lattice,
            # 'loss_coord': loss_coord,
            # 'loss_type': loss_type
        }


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        # log_dict, loss = self.compute_stats(output_dict, prefix='val')
        output_dict['val_loss'] = output_dict['loss_energy']
        self.log_dict(
            output_dict,
            # {'val_loss': output_dict['loss_energy'],},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return output_dict['loss_energy']

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        # log_dict, loss = self.compute_stats(output_dict, prefix='test')
        output_dict['test_loss'] = output_dict['loss_energy']
        self.log_dict(
            output_dict,
            # {'val_loss': output_dict['loss_energy'], },
        )
        return output_dict['loss_energy']

    def compute_stats(self, output_dict, prefix):

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss