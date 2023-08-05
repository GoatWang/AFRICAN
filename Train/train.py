import os
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from Model import AfricanSlowfast
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from DataLoader import MyCollate
from Dataset import AnimalKingdomDataset, AnimalKingdomDatasetSlowFast
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_version = _config['version'] if _config['version'] is not None else datetime_str
    _config['models_dir'] = os.path.join(_config["model_dir"], _config["name"], model_version)
    Path(_config['models_dir']).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(_config["seed"])
    Dataset = AnimalKingdomDatasetSlowFast
    dataset_train = Dataset(_config, split="train")
    dataset_valid = Dataset(_config, split="val")
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

    model = AfricanSlowfast(_config).to(_config['device'])
    dataset_train.produce_prompt_embedding(model.video_clip)
    dataset_valid.produce_prompt_embedding(model.video_clip)
    df_action = dataset_train.df_action
    model.set_class_names(df_action['action'].values)
    model.set_text_feats(dataset_train.text_features)
    model.set_loss_func(_config['loss'], df_action['count'].tolist())
    model.set_metrics(df_action[df_action['segment'] == 'head'].index.tolist(), 
                      df_action[df_action['segment'] == 'middle'].index.tolist(), 
                      df_action[df_action['segment'] == 'tail'].index.tolist())

    collate_fn = MyCollate(_config, model.image_encoder_ic, model.image_encoder_af)
    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"], collate_fn=collate_fn)
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"], collate_fn=collate_fn)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=_config['models_dir'], 
        filename='{epoch}-{valid_MultilabelAveragePrecision:.3f}-{valid_map_head:.3f}-{valid_map_middle:.3f}-{valid_map_tail:.3f}',
        verbose=True,
        save_top_k=3, 
        every_n_epochs=1,
        monitor="valid_MultilabelAveragePrecision", 
        mode="max", 
        save_last=True)
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    csv_logger = pl.loggers.CSVLogger(save_dir=_config["log_dir"], name=_config['name'], version=datetime_str)
    csv_logger.log_hyperparams(_config)
    loggers=[csv_logger]
    
    if _config['wandb']:
        wandb_logger = pl.loggers.WandbLogger(project='AnimalKingdom', save_dir=_config["log_dir"], name=_config['name'], version=model_version)
        wandb_logger.experiment.config.update(_config, allow_val_change=True)
        loggers.append(wandb_logger)

    trainer = pl.Trainer(max_epochs=_config['max_epochs'], 
                        logger=loggers, 
                        log_every_n_steps=(len(dataset_train) // _config['batch_size']) // 3,
                        callbacks=[checkpoint_callback, lr_callback, summary_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=_config['ckpt_path'])

    # optimizer = model.configure_optimizers()
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()