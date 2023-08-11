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

def turn_off_axis_ticks(ax):
    # ax.axis('off')  # Turn off the axis lines
    ax.set_xticks([])  # Turn off the x-axis ticks
    ax.set_yticks([])  # Turn off the y-axis ticks

def plot_attention_map_v2(images_raw, heatmaps_ic, heatmaps_af, fig_fp=None):
    n_rows, n_cols = 3, 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    
    for ci in range(n_cols):
        image_raw, heatmap_ic, heatmap_af = images_raw[ci], heatmaps_ic[ci], heatmaps_af[ci]
        axes[0][ci].imshow(image_raw)
        axes[1][ci].imshow(heatmap_ic)
        axes[2][ci].imshow(heatmap_af)

        axes[0][ci].set_title(f"frame{ci+1}")
        for ri in range(n_rows):
            turn_off_axis_ticks(axes[ri][ci])

    axes[0][0].set_ylabel('Raw')
    axes[1][0].set_ylabel('Clip')
    axes[2][0].set_ylabel('African')
    plt.suptitle("Attention Heatmap")

    if fig_fp:
        plt.savefig(fig_fp)
        plt.close()
    else:
        plt.show()        


@ex.automain
def main(_config):
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
    model.to(_config['device'])
    model.eval()

    for idx in [30, 60, 80, 85, 90, 140, 147, 153]:
        video_fp, video_raw, video_aug, labels_onehot, index = dataset_valid[idx]
        video_raw, video_aug = video_raw.unsqueeze(0), video_aug.unsqueeze(0)
        images_raw, attn_maps, heatmaps_ic = model.draw_image_encoder_att_map(video_raw, video_aug, image_encoder="ic")
        images_raw, attn_maps, heatmaps_af = model.draw_image_encoder_att_map(video_raw, video_aug, image_encoder="af")
        heatmaps_ic, heatmaps_af = heatmaps_ic[0], heatmaps_af[0]

        fig_fp = os.path.join(_config['attn_map_save_dir'], str(idx).zfill(5) + ".png")
        plot_attention_map_v2(images_raw, heatmaps_ic, heatmaps_af, fig_fp=fig_fp)




