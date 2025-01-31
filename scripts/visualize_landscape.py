import json

import hydra
from omegaconf import DictConfig, OmegaConf, ValueNode
from tools.data_utils import (
    preprocess, add_scaled_lattice_prop)
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

import pytorch_lightning as pl
import torch
from typing import List
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.data import Data

from tools.utils import log_hyperparameters, PROJECT_ROOT
from jarvis.db.figshare import data as jdata
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks



def run(cfg:DictConfig):
    # ---- print config ----
    print(OmegaConf.to_yaml(cfg))
    seed_num = 123
    np.random.seed(seed_num)

    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-09-20/10-47-43'}/lc_fr_lr_4"
    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-09-20/11-19-30'}/energy_only"
    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-11/12-42-15'}/lm_fn_ln"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-08-03'}/lc_fr_lrnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-09-39'}/lc_fr_lnnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-11-53'}/lc_fn_lrnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-13-46'}/lc_fn_lnnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-22-54'}/lm_fr_lrnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-16-44'}/lm_fr_lnnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-18-02'}/lm_fn_lrnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-18-36'}/lm_fn_lnnew"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/11-28-06'}/energy_only"

    # mse
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-41-01'}/energy_only"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-45-04'}/lc_fr_lr_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-45-51'}/lc_fr_ln_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-46-35'}/lc_fn_lr_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-47-10'}/lc_fn_ln_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-47-33'}/lm_fr_lr_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/18-48-07'}/lm_fr_ln_emse"
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-11-01/10-59-09'}/lm_fn_lr_emse"  # notice the date
    # "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-11-01/10-57-58'}/lm_fn_ln_emse"  # notice the date


    hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-28/11-28-06'}/energy_only"
    hydra_dir = Path(hydra_dir_str)

    # Instantiate model

    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    lattice_scaler = torch.load( hydra_dir / 'lattice_scaler.pt')
    scaler = torch.load(hydra_dir / 'prop_scaler.pt')
    try:
        model.lattice_scaler = torch.load(hydra_dir / 'lattice_scaler.pt')
        model.scaler = torch.load(hydra_dir / 'prop_scaler.pt')
    except:
        pass

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    hydra.utils.log.info("Instantiating the Trainer")

    model.eval()
    hparams = os.path.join(hydra_dir, "hparams.yaml")
    model = model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=True)

    hydra.utils.log.info("Starting testing!")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('test')
    # test_dataset = hydra.utils.instantiate(cfg.data)   #data=visualize
    chosen_material = None
    # for mpdata in test_dataset.cached_data:
    #     if mpdata['mp_id'] == cfg.data.mpid:
    #         chosen_material = mpdata
    #         break
    # if chosen_material is None:
    #     print('Did not find target material!')
    # X = chosen_material['graph_arrays'][0]
    # D1 = np.random.normal(0, 0.1, X.shape)
    # D2 = np.random.normal(0, 0.1, X.shape)
    # 定义网格的大小
    # grid_size = 20  # 生成20x20的网格
    # energies = np.zeros((grid_size, grid_size))
    # # 计算网格上每个点的能量
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         # 计算 X'(i, j) = X + i * D1 + j * D2
    #         X_prime = X + i * D1 + j * D2
    #         x_new = dict()
    #         for k,v in chosen_material.items():
    #             x_new[k] = v
    #
    #         x_new['graph_arrays'] = (X_prime,*chosen_material['graph_arrays'][1:])
    #         data_dict = x_new
    #         # prop = data_dict[cfg.data.prop]
    #         (frac_coords, atom_types, lengths, angles, edge_indices,
    #          to_jimages, num_atoms) = data_dict['graph_arrays']
    #         material = Data(
    #             frac_coords=torch.Tensor(frac_coords),
    #             atom_types=torch.LongTensor(atom_types),
    #             lengths=torch.Tensor(lengths).view(1, -1),
    #             angles=torch.Tensor(angles).view(1, -1),
    #             edge_index=torch.LongTensor(
    #                 edge_indices.T).contiguous(),  # shape (2, num_edges)
    #             to_jimages=torch.LongTensor(to_jimages),
    #             num_atoms=num_atoms,
    #             num_bonds=edge_indices.shape[0],
    #             num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
    #             # y=prop.view(1, -1),
    #         )
    #
    #         energies[i, j] = model(material)

    test_dataloader = datamodule.test_dataloader()[0]
    energy_list = []
    i = 0
    for batch in test_dataloader:
        result = model(batch)
        energy_list.extend(result['energy'].squeeze().tolist())
        i+=1
        if i%50 == 0:
            print(i)
        pass

    dataset = datamodule.test_datasets[0]
    i_values = dataset.i_values
    j_values = dataset.j_values
    energy_landscape = np.zeros((len(dataset.i_values), len(dataset.j_values)))
    index = 0
    for i_idx, i in enumerate(i_values):
        for j_idx, j in enumerate(j_values):
            # 计算能量并存储在energy_landscape中
            energy_landscape[i_idx, j_idx] = energy_list[index]
            index += 1

    # 2D visualization
    # plt.imshow(energy_landscape, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    # plt.colorbar(label='Energy')
    # plt.xlabel('i')
    # plt.ylabel('j')
    # plt.title('Energy Landscape around X')
    # plt.show()

    # 3D  visualization
    # 创建i, j网格
    I, J = np.meshgrid(i_values, j_values)

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维surface
    surf = ax.plot_surface(I, J, energy_landscape, cmap='viridis')
    fig.colorbar(surf, label='Energy')

    # 添加标签
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('Energy')
    ax.set_title('3D Energy Landscape around X')

    # 保存图片
    mpid = cfg.data.datamodule.datasets.test[0]['mpid']
    sigma = cfg.data.datamodule.datasets.test[0]['sigma']
    plt.savefig(mpid+'_seed_'+str(seed_num)+'_'+hydra_dir_str.split("/")[-1]+'_sigma_'+str(sigma)+'.png', format='png', dpi=300)  # 指定文件名、格式和分辨率
    plt.show()

    # trainer.test(model=model, datamodule=datamodule)
    pass



@hydra.main(config_path= str(PROJECT_ROOT)+"/conf", config_name="default")
def main(cfg: DictConfig):
    run(cfg)

    pass


if __name__ == "__main__":
    print(PROJECT_ROOT)
    main()