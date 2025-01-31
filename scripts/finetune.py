import hydra
from omegaconf import DictConfig, OmegaConf
import os
from tools.data_utils import get_id_train_val_test, save_lists, load_lists
from tools.utils import PROJECT_ROOT
from pytorch_lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from data.mp_dataset import CrystDataset
import pytorch_lightning as pl
import torch
from typing import List
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from tools.utils import log_hyperparameters, PROJECT_ROOT
from jarvis.db.figshare import data as jdata
import pickle
import numpy as np

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

    # finetune on bandgap, change the config
    cfg.data.prop = 'gap pbe'  # 'gap pbe' in megnet dataset
    root_path = cfg.data.root_path
    random_seed = cfg.train.random_seed
    cfg.data.datamodule.datasets.train.save_path = root_path+'/train_bandgap.pt'
    # cfg.data.datamodule.datasets.train.id_list_name = root_path+'_seed'+str(random_seed)+'_bandgap_train.pkl'
    cfg.data.datamodule.datasets.val[0].save_path = root_path + '/val_bandgap.pt'
    # cfg.data.datamodule.datasets.val.id_list_name = root_path + '_seed' + str(random_seed) + '_bandgap_val.pkl'
    cfg.data.datamodule.datasets.test[0].save_path = root_path + '/test_bandgap.pt'
    # cfg.data.datamodule.datasets.test.id_list_name = root_path + '_seed' + str(random_seed) + '_bandgap_test.pkl'


    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    cfg.model.decoder.use_time = True
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    # load from checkpoint
    model_path = Path("/hpai/aios3.0/private/user/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-09-20/11-19-30'}/energy_only")
    ckpts = list(model_path.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt = None
        for ck in ckpts:
            if 'last' in ck.parts[-1]:
                ckpt = str(ck)
        if ckpt is None:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts if 'last' not in ckpt.parts[-1]])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    hparams = os.path.join(model_path, "hparams.yaml")
    model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=False)

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    wandb.login(key="29e839bd27fe2c8a0058a0e0cf94f8d9e39b7a93")
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

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
    trainer = pl.Trainer(
        default_root_dir=hydra_dir.__str__(),
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)
    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    pass


@hydra.main(config_path= str(PROJECT_ROOT)+"/conf", config_name="default")
def main(cfg: DictConfig):
    run(cfg)

    pass


if __name__ == "__main__":
    print(PROJECT_ROOT)
    main()