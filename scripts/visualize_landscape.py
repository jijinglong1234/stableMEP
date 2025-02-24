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

    hydra_dir_str = ""
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
           
            energy_landscape[i_idx, j_idx] = energy_list[index]
            index += 1
    I, J = np.meshgrid(i_values, j_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(I, J, energy_landscape, cmap='viridis')
    fig.colorbar(surf, label='Energy')

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('Energy')
    ax.set_title('3D Energy Landscape around X')

    mpid = cfg.data.datamodule.datasets.test[0]['mpid']
    sigma = cfg.data.datamodule.datasets.test[0]['sigma']
    plt.savefig(mpid+'_seed_'+str(seed_num)+'_'+hydra_dir_str.split("/")[-1]+'_sigma_'+str(sigma)+'.png', format='png', dpi=300)  
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
