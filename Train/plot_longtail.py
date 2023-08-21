import os
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from config import ex, config
from datetime import datetime
import pytorch_lightning as pl
from Model import AfricanSlowfast
from matplotlib import pyplot as plt
from Dataset import AnimalKingdomDatasetVisualize
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    Path(_config['attn_map_save_dir']).mkdir(exist_ok=True, parents=True)
    pl.seed_everything(_config["seed"])

    dataset_train = AnimalKingdomDatasetVisualize(_config, split="train")
    dataset_valid = AnimalKingdomDatasetVisualize(_config, split="val")
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

    # df_action['action'].values
    df_action = dataset_train.df_action
    df_action.loc[df_action['segment'] == 'head', 'color'] = 'green'
    df_action.loc[df_action['segment'] == 'middle', 'color'] = 'blue'
    df_action.loc[df_action['segment'] == 'tail', 'color'] = 'red'

    # df_action.sort_values('count')[['action', 'count']].plot()
    df_action_sorted = df_action.sort_values('count', ascending=False)

    plt.figure(figsize=(12, 5))
    plt.bar(x=range(len(df_action_sorted)), height=df_action_sorted['count'], color=df_action_sorted['color'])
    plt.ylabel('Count', size=12)
    plt.xlabel('Action ID', size=12)
    plt.title("Action Class Frequency", size=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), "temp", "LongTail.png"))
