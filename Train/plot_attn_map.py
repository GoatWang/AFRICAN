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

def plot_attention_map_v2(images_raw, heatmaps_vc, heatmaps_ic, heatmaps_af, fig_fp=None):
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    
    for ci in range(n_cols):
        image_raw, heatmap_vc, heatmap_ic, heatmap_af = images_raw[ci], heatmaps_vc[ci], heatmaps_ic[ci], heatmaps_af[ci]
        axes[0][ci].imshow(image_raw)
        # axes[1][ci].imshow(heatmap_vc)
        axes[1][ci].imshow(heatmap_ic)
        axes[2][ci].imshow(heatmap_af)

        axes[0][ci].set_title(f"frame{ci+1}")
        for ri in range(n_rows):
            turn_off_axis_ticks(axes[ri][ci])

    axes[0][0].set_ylabel('Raw Image', size=14)
    # axes[1][0].set_ylabel('VideoClip')
    axes[1][0].set_ylabel('IC', size=14)
    axes[2][0].set_ylabel('AFRICAN', size=14)
    # plt.suptitle("Attention Heatmap")

    plt.tight_layout()
    if fig_fp:
        plt.savefig(fig_fp)
        plt.close()
    else:
        plt.show()        


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    Path(_config['attn_map_save_dir']).mkdir(exist_ok=True, parents=True)
    Path(_config['attn_map_save_dir'] + "_selected").mkdir(exist_ok=True, parents=True)
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

    def inference_on_idx(idx, save_dir):
        video_fp, video_raw, video_aug, labels_onehot, index = dataset_valid[idx]
        video_raw, video_aug = video_raw.unsqueeze(0), video_aug.unsqueeze(0)
        images_raw, attn_maps, heatmaps_vc = model.draw_att_map(video_raw, video_aug, encoder_type="vc") # heatmaps_vc.shape = []
        images_raw, attn_maps, heatmaps_ic = model.draw_att_map(video_raw, video_aug, encoder_type="ic") # heatmaps_ic.shape = []
        images_raw, attn_maps, heatmaps_af = model.draw_att_map(video_raw, video_aug, encoder_type="af") # heatmaps_af.shape = []
        images_raw, heatmaps_vc, heatmaps_ic, heatmaps_af = images_raw[0], heatmaps_vc[0], heatmaps_ic[0], heatmaps_af[0]
        fig_fn = os.path.basename(video_fp).split('.')[0] + ".png"
        fig_fp = os.path.join(save_dir, fig_fn)
        plot_attention_map_v2(images_raw, heatmaps_vc, heatmaps_ic, heatmaps_af, fig_fp=fig_fp)
        print("file saved to", fig_fp)
        
    # for idx in [30, 60, 80, 85, 90, 140, 147, 153]:
    targets = ["AQFMKMRN", "ADRIOBSK", "AGXDIPKK", "AJCYBKDQ", "AJJBHNQN", "AKIEHTKX", "AORAGPTK", "APINQPKK"]
    target_idxs = [idx for idx, fp in enumerate(dataset_valid.video_fps) if os.path.basename(fp).split(".")[0] in targets]
    for idx in target_idxs:
        save_dir = _config['attn_map_save_dir'] + "_selected"
        inference_on_idx(idx, save_dir)

    for idx in np.random.choice(range(len(dataset_valid)), 200):
        save_dir = _config['attn_map_save_dir']
        inference_on_idx(idx, save_dir)


# !python3 /notebooks/AnimalKingdomCLIP/Train/plot_attn_map.py with \
# 'data_dir="/storage/AnimalKingdom/action_recognition"' \
# 'device="cuda"' 'enable_video_clip=False' 'enable_image_clip=True' 'enable_african=True'

# %cd /notebooks/AnimalKingdomCLIP/Train/temp
# !tar -cvzf attn_map.tar.gz attn_map
# %cd /notebooks
