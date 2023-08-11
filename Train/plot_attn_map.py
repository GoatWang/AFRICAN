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
from Dataset import AnimalKingdomDatasetVisualize
torch.manual_seed(0)

# @ex.automain
# def main(_config):

if __name__ == '__main__':
    _config = config()
    _config = copy.deepcopy(_config)
    Path(_config['attn_map_save_dir']).mkdir(exist_ok=True, parents=True)
    pl.seed_everything(_config["seed"])

    dataset_train = AnimalKingdomDatasetVisualize(_config, split="train")
    dataset_valid = AnimalKingdomDatasetVisualize(_config, split="val")
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

    _config['batch_size'] = 1
    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"])
    # valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"])

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            model.visualize_attn_map(batch, _config['attn_map_save_dir'])
            break


