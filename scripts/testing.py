import hydra
from omegaconf import DictConfig, OmegaConf

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
import os

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

    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-09-20/10-47-43'}/lc_fr_lr_4"
    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-09-20/11-19-30'}/energy_only"
    # hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-11/12-42-15'}/lm_fn_ln"
    # mae
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

    hydra_dir_str = "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-11-01/10-57-58'}/lm_fn_ln_emse"

    # hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir = Path(hydra_dir_str)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False,scaler_path = hydra_dir
    )

    # Instantiate model
    # cfg.model._target_ = 'model.pl_modules.energy_model.Matformer_pl'

    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    lattice_scaler = torch.load( hydra_dir / 'lattice_scaler.pt')
    scaler = torch.load(hydra_dir / 'prop_scaler.pt')
    try:
        model.lattice_scaler = torch.load(hydra_dir / 'lattice_scaler.pt')
        model.scaler = torch.load(hydra_dir / 'prop_scaler.pt')
    except:
        pass
    if datamodule.scaler is not None:
        model.lattice_scaler = lattice_scaler
        model.scaler = scaler
    # torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    # torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
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


    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir.__str__(),
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        # resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )
    model.eval()
    # print("Before model.decoder.node_embedding.weight.data:",model.decoder.node_embedding.weight.data)
    # state_dict_before = model.state_dict()

    # model = model.load_from_checkpoint(ckpt)

    hparams = os.path.join(hydra_dir, "hparams.yaml")
    model = model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=True)

    # print("After model.decoder.node_embedding.weight.data",model.decoder.node_embedding.weight.data)
    # state_dict_after = model.state_dict()
    # for key in state_dict_before.keys():
    #     if not torch.equal(state_dict_before[key], state_dict_after[key]):
    #         print(f"Parameter {key} has changed.")
    # log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)
    pass



@hydra.main(config_path= str(PROJECT_ROOT)+"/conf", config_name="default")
def main(cfg: DictConfig):
    run(cfg)

    pass


if __name__ == "__main__":
    print(PROJECT_ROOT)
    main()