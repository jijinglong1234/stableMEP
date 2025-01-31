import json
import os

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
from jarvis.db.figshare import data as jdata, get_request_data
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
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # js_tag = 'MPtrj_2022_9_full.json'
    # js_tag = 'm3gnet_mpf.json'
    # path = str(os.path.join(
    #     '/home/huangjiao/anaconda3/envs/diffcsp2/lib/python3.8/site-packages/jarvis/db', js_tag))
    # with open(path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # data_list = []
    # for k, v in data.items():
    #     for k1, v1 in v.items():
    #         v1['mptrj_id'] = k1
    #         data_list.append(v1)
    # dat = data_list

    # scaler = torch.load(Path(
    #     "/home/huangjiao/codes/stableMPP/outputs/hydra/singlerun/2024-10-09/11-18-36'}/lm_fn_lnnew") / 'prop_scaler.pt')
    # one_std_list = []
    # two_std_list = []
    # three_std_list = []
    # more_std_list = []

    # for i in range(len(data_list)):
    #     material = data_list[i]
    #     if abs(material['ef_per_atom'] - scaler.means) < abs(scaler.stds):
    #         one_std_list.append(i)
    #     elif abs(material['ef_per_atom'] - scaler.means) < 2*abs(scaler.stds):
    #         two_std_list.append(i)
    #     elif abs(material['ef_per_atom'] - scaler.means) < 3*abs(scaler.stds):
    #         three_std_list.append(i)
    #     else:
    #         more_std_list.append(i)

    # data = jdata('megnet')
    # id_train, id_val, id_test = get_id_train_val_test(
    #     # total_size=len(data),
    #     # total_size=1580395,
    #     total_size=cfg.data.total_size,
    #     split_seed=cfg.train.random_seed,
    #     train_ratio=cfg.data.train_ratio,
    #     val_ratio=cfg.data.val_ratio,
    #     test_ratio=cfg.data.test_ratio,
    #     n_train=cfg.data.n_train,
    #     n_test=cfg.data.n_test,
    #     n_val=cfg.data.n_val,
    #     keep_data_order=cfg.data.keep_data_order,
    # )
    # path = cfg.data.root_path + '/seed_' + str(cfg.train.random_seed) + '_numtest'
    # filenames = [path + '_train.pkl', path + '_val.pkl', path + '_test.pkl']

    # save_lists([id_train, id_val, id_test], filenames)
    # loaded_lists = load_lists(filenames)

    # Instantiate datamodule
    # cfg.data.datamodule.datasets.train.save_path = cfg.data.root_path+r'/train_ori_matformer.pt'
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    # train_dataset = hydra.utils.instantiate(cfg.data.datamodule.datasets.train)



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

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # name = 'Formation energy train'
    # prop = 'e_form'
    # save_path = '/home/huangjiao/codes/stableMPP/data/megnet/train_ori.pt'
    # niggli = True
    # primitive = False
    # graph_method = 'crystalnn'
    # tolerance = 0.1
    # use_space_group = False
    # use_pos_index = False
    # lattice_scale_method = 'scale_length'
    # preprocess_workers = 30
    # dataset_name = 'megnet'
    #
    #
    # train_dataset = CrystDataset(name,  prop, niggli, primitive, graph_method, preprocess_workers,
    #              lattice_scale_method, save_path, tolerance, use_space_group, use_pos_index, dataset_name)
    pass


@hydra.main(config_path= str(PROJECT_ROOT)+"/conf", config_name="default")
def main(cfg: DictConfig):
    run(cfg)

    pass


if __name__ == "__main__":
    print(PROJECT_ROOT)
    main()