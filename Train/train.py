import os
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from Model import VideoCLIP
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from Dataset import AnimalKingdomDataset
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    _config['version'] = datetime.now().strftime("%Y%m%d-%H%M%S")
    _config['models_dir'] = os.path.join(_config["model_dir"], _config["name"], _config['version'])
    Path(_config['models_dir']).mkdir(parents=True, exist_ok=True)

    model = VideoCLIP(_config).to(_config['device'])
    pl.seed_everything(_config["seed"])
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)
    model.set_class_names(dataset_train.df_action['action'].values)
    model.set_text_feats(dataset_train.text_features)
    print("train baseline (140 classes):", (1 - np.bincount(np.hstack(dataset_train.labels)) / len(dataset_train)))
    print("valid baseline (140 classes):", (1 - np.bincount(np.hstack(dataset_valid.labels)) / len(dataset_valid)))

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"]) # bugs on MACOS
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"]) # bugs on MACOS

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=_config['models_dir'], verbose=True) # 
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=_config['models_dir'], 
        verbose=True,
        save_top_k=3, 
        every_n_epochs=1,
        monitor="train_loss", 
        mode="max", 
        save_last=True)
    # lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)

    csv_logger = pl.loggers.CSVLogger(save_dir=_config["log_dir"], name=_config['name'], version=_config['version'])
    csv_logger.log_hyperparams(_config)
    # wandb_logger = pl.loggers.WandbLogger(project='AnimalKingdom', save_dir=_config["log_dir"], name=_config['name'], version=_config['version'])
    # wandb_logger.experiment.config.update(_config)
    trainer = pl.Trainer(max_epochs=_config['max_epochs'], 
                        logger=csv_logger, 
                        #  logger=wandb_logger, 
                        callbacks=[checkpoint_callback, summary_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=_config['animal_kingdom_clip_path'])

    # optimizer = model.configure_optimizers()
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()