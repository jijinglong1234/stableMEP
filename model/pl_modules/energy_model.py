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


class CSPEnergy(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim, pred_scalar = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)

        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type      


    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]



        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        if self.update_type:
            atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        else:
            atom_type_probs = gt_atom_types_onehot


        pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        loss_energy = F.l1_loss(pred_e, batch.y)


        loss = loss_energy

        return {
            'loss' : loss,
        }

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio = 1.0, step_lr = 1e-5, aug = 1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type


        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()


        
        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            if self.hparams.latent_dim > 0:            
                time_emb = torch.cat([time_emb, z], dim = -1)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t
            
            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T


            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)


                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1

                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = t_T


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):
        
        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim = -1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta



    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim = -1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat, 
            target=theta, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):
        
        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:,types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)



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

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss


class CSPEnergy_denoising(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.hparams.denoising is True:
            self.hparams.decoder._target_ = 'model.pl_modules.cspnet.CSPNet_outputall'
        self.pred_other_scalar = self.hparams.pred_other_scalar
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim= self.hparams.latent_dim + self.hparams.time_dim,
                                               pred_type = True, pred_scalar=True, smooth=True, pred_other_scalar=self.pred_other_scalar)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.lattice_noise = self.hparams.lattice_noise  # 'Riemann' or 'normal'
        self.frac_noise = self.hparams.frac_noise  # 'Riemann' or 'normal'
        self.loss_type = self.hparams.loss_type  # 'MSE' or 'cosine'

        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        times0 = torch.zeros(batch_size).to('cuda')
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
        conc_time_emb = torch.cat((time_emb,time0_emb),dim = 0)
        conc_atom_type_probs = torch.cat((atom_type_probs_noised, gt_atom_types_onehot), dim=0)
        conc_frac_coords = torch.cat((input_frac_coords, frac_coords), dim=0)
        conc_input_lattice = torch.cat((input_lattice, lattices), dim=0)
        batch_num_atoms = torch.cat((batch.num_atoms,batch.num_atoms),dim=0)
        batch_batch = torch.cat((batch.batch,(batch.batch+batch_size)),dim=0)


        # if self.hparams.denoising is True:
        with torch.enable_grad():
            # with RequiresGradContext(atom_type_probs, input_frac_coords, input_lattice, gt_atom_types_onehot, frac_coords,
            #                                                   lattices, requires_grad=True):
            with RequiresGradContext(conc_atom_type_probs, conc_frac_coords, conc_input_lattice, requires_grad=True):
                # pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms,
                #                       batch.batch)

                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # stime2 = time.time()
                pred_l, pred_x, pred_t, pred_e = self.decoder(conc_time_emb, conc_atom_type_probs, conc_frac_coords, conc_input_lattice, batch_num_atoms,
                                                                  batch_batch)
                grad_outputs = [torch.ones_like(pred_e)]
                # grad_t, grad_x, grad_l = grad(pred_e, [conc_atom_type_probs, conc_frac_coords, conc_input_lattice],
                #                               grad_outputs=grad_outputs, allow_unused=True, retain_graph=True,create_graph=True)
                grad_x, grad_l = grad(pred_e, [conc_frac_coords, conc_input_lattice],
                                              grad_outputs=grad_outputs, allow_unused=True, retain_graph=True,
                                              create_graph=True)
                # print(prof.key_averages().table(sort_by="cuda_time_total"))

        # calculate the grad of noised lattice
        if self.lattice_noise == 'Riemann':
            term1 = torch.matmul(input_lattice.transpose(1, 2), input_lattice)  # (L̃^T L̃), shape: (32, 3, 3)
            term2 = torch.matmul(lattices.transpose(1, 2), lattices)  # (L^T L), shape: (32, 3, 3)
            tar_gradl = -1 /  torch.matmul(term1 - term2, input_lattice) #sigma can be ignored because of the loss function
        elif self.lattice_noise == 'normal':
            tar_gradl = rand_l
        else:
            raise ValueError("lattice_noise type is not defined", self.lattice_noise)

        #calculate the grad of noised frac coords
        if self.frac_noise == 'Riemann':
            epsilon = input_frac_coords - frac_coords
            y_bar = torch.mean(torch.sin(2 * np.pi * epsilon), dim=0) # Calculate y̅(𝜖)
            x_bar = torch.mean(torch.cos(2 * np.pi * epsilon), dim=0) # Calculate and x̅(𝜖)
            epsilon_bar = torch.fmod(epsilon-torch.atan2(y_bar, x_bar) / (2 * np.pi),1) # Calculate epsilon̅
            tar_gradf = -2 * np.pi * torch.sin(2 * np.pi * epsilon_bar) # Compute -2π * sin(2π * epsilon̅), k can be ignored because of the loss function
        elif self.frac_noise == 'normal':
            tar_gradf = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        else:
            raise ValueError("frac_noise type is not defined", self.frac_noise)


        # concat label: tarA, tarX, tarL, tarE
        # tar_type = torch.cat((rand_t,torch.zeros_like(rand_t)),dim=0)
        # tar_x = torch.cat((tar_gradf,torch.zeros_like(grad_x)),dim=0)
        # tar_l = torch.cat((tar_gradl,torch.zeros_like(grad_l)),dim=0)
        # tar_e = torch.cat((tar_gradl,torch.zeros_like(grad_l)),dim=0)

        # loss of noised data
        if self.loss_type == 'MSE':
            loss_lattice = F.mse_loss(pred_l[:batch_size], tar_gradl)
            loss_lattice2 = F.mse_loss(grad_l[:batch_size], tar_gradl)
            loss_coord = F.mse_loss(pred_x[:num_atoms], tar_gradf)
            loss_coord2 = F.mse_loss(grad_x[:num_atoms], tar_gradf)
        elif self.loss_type == 'cosine':
            loss_lattice = (1 - F.cosine_similarity(pred_l[:batch_size], tar_gradl, dim=1)).mean()
            loss_lattice2 = (1 - F.cosine_similarity(grad_l[:batch_size], tar_gradl, dim=1)).mean()
            loss_coord = (1 - F.cosine_similarity(pred_x[:num_atoms], tar_gradf, dim=1)).mean()
            loss_coord2 = (1 - F.cosine_similarity(grad_x[:num_atoms], tar_gradf, dim=1)).mean()
        else:
            raise ValueError("loss_type type is not defined", self.loss_type)
        # loss_lattice2 = F.mse_loss(grad_nl, tar_gradl)

        # loss_coord = F.mse_loss(pred_x[:num_atoms], tar_gradf)
        # loss_coord2 = F.mse_loss(grad_nx, tar_gradf)
        loss_type = F.mse_loss(pred_t[:num_atoms], rand_t)
        loss_noised_total = loss_lattice + loss_lattice2 + loss_coord + loss_coord2

        # loss of unnoised data
        # loss of energy
        loss_energy = F.l1_loss(pred_e[batch_size:], batch.y)
        # precompute zeros to reduce redundant operations
        zero_l = torch.zeros_like(pred_l[batch_size:], device=pred_l.device)
        # zero_grad_l = torch.zeros_like(grad_l[batch_size:], device=grad_l.device)
        zero_x = torch.zeros_like(pred_x[num_atoms:], device=pred_x.device)
        # zero_grad_x = torch.zeros_like(grad_x[num_atoms:], device=grad_x.device)
        loss_lattice_unn = F.mse_loss(pred_l[batch_size:], zero_l)
        loss_lattice2_unn = F.mse_loss(grad_l[batch_size:], zero_l)
        loss_coord_unn = F.mse_loss(pred_x[num_atoms:], zero_x)
        loss_coord2_unn = F.mse_loss(grad_x[num_atoms:], zero_x)
        loss_unnoised_total = loss_energy + loss_lattice_unn + loss_lattice2_unn + loss_coord_unn + loss_coord2_unn

        # tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        # loss_lattice = F.mse_loss(pred_nl, rand_l)
        # loss_coord = F.mse_loss(pred_nx, rand_x)
        # loss_type = F.mse_loss(pred_nt, rand_t)
        # else:
        #     pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        loss = loss_unnoised_total

        if self.hparams.denoising is True:
            # loss = (loss_energy +
            #         self.hparams.cost_lattice * loss_lattice +
            #         self.hparams.cost_coord * loss_coord +
            #         self.hparams.cost_type * loss_type)
            loss = loss_noised_total + loss_unnoised_total

        return {
            'loss': loss,
            'loss_energy': loss_energy,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_type': loss_type,
            'loss_lattice2': loss_lattice2,
            'loss_coord2': loss_coord2,
            'loss_noised_total': loss_noised_total,
            'loss_lattice_unn': loss_lattice_unn,
            'loss_lattice2_unn': loss_lattice2_unn,
            'loss_coord_unn': loss_coord_unn,
            'loss_coord2_unn': loss_coord2_unn,
            'loss_unnoised_total': loss_unnoised_total,
        }

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio=1.0, step_lr=1e-5, aug=1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type

        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(
            batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': t_T,
            'frac_coords': x_T % 1.,
            'lattices': l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size,), t, device=self.device)

            time_emb = self.time_embedding(times)

            if self.hparams.latent_dim > 0:
                time_emb = torch.cat([time_emb, z], dim=-1)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05,
                                                       batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms,
                                              batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05],
                                                      grad_outputs=grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1

                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms,
                                               batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs=grad_outputs,
                                              allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l

                t_t_minus_1 = t_T

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': t_t_minus_1,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):

        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim=-1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                    alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta

    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim=-1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                    alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                    alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat,
            target=theta,
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):

        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:, types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)

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

class CSPPretrain(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.hparams.denoising is True:
            self.hparams.decoder._target_ = 'model.pl_modules.cspnet.CSPNet_outputall'
        self.pred_other_scalar = self.hparams.pred_other_scalar
        self.decoder = hydra.utils.instantiate(self.hparams.decoder,
                                               latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
                                               pred_type=True, pred_scalar=True, smooth=True,
                                               pred_other_scalar=self.pred_other_scalar)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.lattice_noise = self.hparams.lattice_noise  # 'Riemann' or 'normal'
        self.frac_noise = self.hparams.frac_noise  # 'Riemann' or 'normal'
        self.loss_type = self.hparams.loss_type  # 'MSE' or 'cosine'
        self.energy_only = self.hparams.energy_only # True or False

        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type

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
        # input_lattice = lattices + sigmas.view(-1, 1, 1) * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

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

        with torch.enable_grad():
            with RequiresGradContext(conc_frac_coords, conc_input_lattice, requires_grad=True):
            # with RequiresGradContext(conc_atom_type_probs, conc_frac_coords, conc_input_lattice, requires_grad=True):
                pred_l, pred_x, pred_t, pred_e = self.decoder(conc_time_emb, conc_atom_type_probs, conc_frac_coords,
                                                              conc_input_lattice, batch_num_atoms,
                                                              batch_batch)

                grad_outputs = [torch.ones_like(pred_e)]
                if not self.energy_only:
                    grad_x, grad_l = grad(pred_e, [conc_frac_coords, conc_input_lattice],grad_outputs=grad_outputs,
                                          allow_unused=True, retain_graph=True, create_graph=True)
                else:
                    grad_x, grad_l = grad(pred_e, [conc_frac_coords, conc_input_lattice], grad_outputs=grad_outputs,
                                      allow_unused=True, retain_graph=True)

        # torch.cuda.empty_cache()
        # calculate the grad of noised lattice

        if self.lattice_noise == 'Riemann':
            term1 = torch.matmul(input_lattice.transpose(1, 2), input_lattice)  # (L̃^T L̃), shape: (N, 3, 3)
            term2 = torch.matmul(lattices.transpose(1, 2), lattices)  # (L^T L), shape: (N, 3, 3)
            tar_gradl = -1 * torch.matmul(term1 - term2, input_lattice) # - (L̃^T L̃-L^T L)L̃
            tar_gradl = 2 * tar_gradl / (torch.norm(term1, dim=(1, 2)) + torch.norm(term2, dim=(1, 2))).view(-1, 1, 1)  # -(L̃^T L̃-L^T L)L̃/α α=(||L̃^T L̃||+||L^T L||)/2
            tar_gradl = tar_gradl / sigmas.view(-1, 1, 1)  # div sigma instead of sigma**2, because of the weights of F(x) in (12) of 3D-EMGP  /sigma**2 * sigma=/sigma
        elif self.lattice_noise == 'normal':
            # tar_gradl = rand_l
            tar_gradl = -1 * rand_l  # -1 * (l+sigma*randl - l)/sigma**2 * sigma=-1 * sigma*rand_l/sigma = -1 * rand_l
        else:
            raise ValueError("lattice_noise type is not defined", self.lattice_noise)

        # calculate the grad of noised frac coords
        if self.frac_noise == 'Riemann':
            # data_tensor = (sigmas_per_atom * rand_x) % 1.0
            # groups = batch.batch
            # epsilon = (input_frac_coords - frac_coords)%1.0
            # y_bar = scatter_mean(torch.sin(2 * np.pi * epsilon),batch.batch, dim=-2)  # Calculate y̅(𝜖)
            # x_bar = scatter_mean(torch.cos(2 * np.pi * epsilon),batch.batch, dim=-2)  # Calculate and x̅(𝜖)
            # epsilon_bar = torch.fmod(epsilon - (torch.atan2(y_bar, x_bar)%(2*np.pi)/(2*np.pi))[batch.batch] , 1)  # Calculate epsilon̅
            # tar_gradf_old = -2 * np.pi * torch.sin(
            #     2 * np.pi * epsilon_bar)  # Compute -2π * sin(2π * epsilon̅), k can be ignored because of the loss function

            invarance_reference = calc_grouped_angles_mean_in_radians((sigmas_per_atom * rand_x) % 1.0, batch.batch)
            invarance_reference = invarance_reference.repeat_interleave(batch.num_atoms, dim=0)
            invariance_noise = (sigmas_per_atom * rand_x - invarance_reference) % 1.

            if self.loss_type == 'MSE':
                kappa = self.sigma_scheduler.kappa_matrix[batch.num_atoms, times]
                kappa_norm = self.sigma_scheduler.kappa_norm[batch.num_atoms, times]
                kappa_per_atom = kappa.repeat_interleave(batch.num_atoms)[:, None]
                kappa_norm_per_atom = kappa_norm.repeat_interleave(batch.num_atoms)[:, None]

                tar_score = kappa_per_atom * torch.sin(invariance_noise * 2 * math.pi) * 2 * math.pi
                tar_gradf = tar_score / torch.sqrt(kappa_norm_per_atom)
            elif self.loss_type == 'cosine':
                tar_gradf = torch.sin(invariance_noise * 2 * math.pi) * 2 * math.pi

        elif self.frac_noise == 'normal':
            tar_gradf = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(
                sigmas_norm_per_atom)
        else:
            raise ValueError("frac_noise type is not defined", self.frac_noise)

        # visualize
        # from tools.visualize_crystal import visualize_crystal
        #
        # idx_tmp = 0
        # for i in range(len(batch)):
        #     crystal1 = batch.to_data_list()[i]
        #     visualize_crystal(crystal1.atom_types.cpu().numpy(), input_lattice[i].cpu().numpy(),
        #                       crystal1.frac_coords.cpu().numpy(), tar_gradf[:batch.num_atoms[i]].cpu().numpy())
        #
        #     visualize_crystal(torch.cat((crystal1.atom_types,crystal1.atom_types)).cpu().numpy(), input_lattice[i].cpu().numpy()/lattices[i].cpu().numpy(),
        #                       torch.cat((input_frac_coords[idx_tmp:idx_tmp+batch.num_atoms[i]],frac_coords[idx_tmp:idx_tmp+batch.num_atoms[i]])).cpu().numpy(),
        #                       torch.cat((tar_gradf[idx_tmp:idx_tmp+batch.num_atoms[i]],
        #                                  torch.zeros_like(tar_gradf[idx_tmp:idx_tmp+batch.num_atoms[i]]))).cpu().numpy())
        #     idx_tmp += batch.num_atoms[i]

        # loss of noised data

        if self.loss_type == 'MSE':
            loss_lattice = F.mse_loss(pred_l[:batch_size], tar_gradl)
            loss_lattice2 = F.mse_loss(grad_l[:batch_size], tar_gradl)
            loss_coord = F.mse_loss(pred_x[:num_atoms], tar_gradf)
            loss_coord2 = F.mse_loss(grad_x[:num_atoms], tar_gradf)
        elif self.loss_type == 'cosine':
            loss_lattice = (1 - F.cosine_similarity(pred_l[:batch_size], tar_gradl, dim=1)).mean()
            loss_lattice2 = (1 - F.cosine_similarity(grad_l[:batch_size], tar_gradl, dim=1)).mean()
            loss_coord = (1 - F.cosine_similarity(pred_x[:num_atoms], tar_gradf, dim=1)).mean()
            loss_coord2 = (1 - F.cosine_similarity(grad_x[:num_atoms], tar_gradf, dim=1)).mean()
        else:
            raise ValueError("loss_type type is not defined", self.loss_type)
        # print(loss_lattice,loss_lattice2,loss_coord,loss_coord2)
        loss_type = F.mse_loss(pred_t[:num_atoms], rand_t)
        loss_noised_total = loss_lattice + loss_lattice2 + loss_coord + loss_coord2

        # loss of unnoised data
        # loss of energy
        loss_energy = F.l1_loss(pred_e[batch_size:], batch.y)
        # loss_energy = F.mse_loss(pred_e[batch_size:], batch.y)
        loss_energy_mae = F.l1_loss(pred_e[batch_size:], batch.y)
        # precompute zeros to reduce redundant operations
        zero_l = torch.zeros_like(pred_l[batch_size:], device=pred_l.device)
        zero_x = torch.zeros_like(pred_x[num_atoms:], device=pred_x.device)
        # loss funcs
        loss_lattice_unn = F.mse_loss(pred_l[batch_size:], zero_l)
        loss_lattice2_unn = F.mse_loss(grad_l[batch_size:], zero_l)
        loss_coord_unn = F.mse_loss(pred_x[num_atoms:], zero_x)
        loss_coord2_unn = F.mse_loss(grad_x[num_atoms:], zero_x)
        loss_unnoised_total = loss_energy + loss_lattice_unn + loss_lattice2_unn + loss_coord_unn + loss_coord2_unn

        loss = loss_noised_total + loss_unnoised_total
        if self.energy_only:
            loss = loss_energy

        return {
            'loss': loss,
            'loss_energy': loss_energy,
            'loss_energy_mae': loss_energy_mae,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_type': loss_type,
            'loss_lattice2': loss_lattice2,
            'loss_coord2': loss_coord2,
            'loss_noised_total': loss_noised_total,
            'loss_lattice_unn': loss_lattice_unn,
            'loss_lattice2_unn': loss_lattice2_unn,
            'loss_coord_unn': loss_coord_unn,
            'loss_coord2_unn': loss_coord2_unn,
            'loss_unnoised_total': loss_unnoised_total,
            'energy': pred_e[batch_size:].detach(),
            'force': grad_x.detach(),
            'stress': grad_l.detach(),
        }

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio=1.0, step_lr=1e-5, aug=1.0):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type

        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(
            batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': t_T,
            'frac_coords': x_T % 1.,
            'lattices': l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size,), t, device=self.device)

            time_emb = self.time_embedding(times)

            if self.hparams.latent_dim > 0:
                time_emb = torch.cat([time_emb, z], dim=-1)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                pred_x = pred_x * torch.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            if update_type:

                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05,
                                                       batch.num_atoms, batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms,
                                              batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05],
                                                      grad_outputs=grad_outputs, allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:

                t_t_one = t_T.argmax(dim=-1).long() + 1

                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms,
                                               batch.batch)

                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs=grad_outputs,
                                              allow_unused=True)

                pred_x = pred_x * torch.sqrt(sigma_norm)

                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l

                t_t_minus_1 = t_T

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': t_t_minus_1,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):

        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim=-1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta

    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim=-1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (
                alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat,
            target=theta,
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):

        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:, types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try:
            output_dict = self(batch)
            loss = output_dict['loss']
            # torch.autograd.backward(loss,retain_graph=True)
            self.log_dict(
                {'train_loss': loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            if loss.isnan():
                return None
            return loss
        except BaseException as e:
            if 'out of memory' in str(e):
                print(f"Out of memory error during loss.backward(): skipping batch {batch_idx}")
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def training_epoch_end(self, outputs):
        # 处理 None 的情况，移除 None 值，防止影响后续逻辑
        outputs[:] = [o for o in outputs if o is not None]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try:
            output_dict = self(batch)

            # log_dict, loss = self.compute_stats(output_dict, prefix='val')
            output_dict['val_loss'] = output_dict['loss_energy_mae']
            self.log_dict(
                output_dict,
                # {'val_loss': output_dict['loss_energy'],},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return output_dict['loss_energy_mae']
        except BaseException as e:
            if 'out of memory' in str(e):
                print(f"Out of memory error during loss.backward(): skipping batch {batch_idx}")
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def validation_epoch_end(self, outputs):
        # 移除 None 值
        outputs = [o for o in outputs if o is not None]
        if len(outputs) > 0:
            avg_loss = torch.stack(outputs).mean()
            self.log('val_loss_epoch', avg_loss, prog_bar=True)
        else:
            print("No valid outputs for this epoch, all batches were skipped.")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try:
            output_dict = self(batch)

            # log_dict, loss = self.compute_stats(output_dict, prefix='test')
            output_dict['test_loss'] = output_dict['loss_energy_mae']
            self.log_dict(
                output_dict,
                # {'val_loss': output_dict['loss_energy'], },
            )
            return output_dict['loss_energy_mae']
        except BaseException as e:
            if 'out of memory' in str(e):
                print(f"Out of memory error during forward(): skipping batch {batch_idx}")
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def test_epoch_end(self, outputs):
        # 移除 None 值
        outputs = [o for o in outputs if o is not None]
        if len(outputs) > 0:
            avg_loss = torch.stack(outputs).mean()
            self.log('test_loss_epoch', avg_loss, prog_bar=True)
        else:
            print("No test outputs for this epoch, all batches were skipped.")

    def compute_stats(self, output_dict, prefix):

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss

    def backward(self, loss, optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        if self.automatic_optimization or self._running_manual_backward:
            try:
                loss.backward(*args, **kwargs)
            except BaseException as e:
                if 'out of memory' in str(e):
                    print(f"Out of memory error during loss.backward()")
                    torch.cuda.empty_cache()

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, *args, **kwargs):
    #     if not getattr(self, 'skip_backward', False):
    #         optimizer.step(closure=second_order_closure)
    #     else:
    #         print(f"Skipping optimizer step at batch {batch_idx} due to OOM")
    #         self.skip_backward = False


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

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

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
            atom_type_probs = (
                        c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(
                    batch.num_atoms)[:, None] * rand_t)
        else:
            atom_type_probs = gt_atom_types_onehot

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