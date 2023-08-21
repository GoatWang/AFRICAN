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
from sklearn.decomposition import PCA
from Dataset import AnimalKingdomDatasetVisualize

def check_and_adjust_overlap(text_objects, colors, n_iters=30, threshold=0.1, seed=2023):
    np.random.seed(seed)
    
    legend_idxs = []
    # Iterate until no overlaps are found
    for i in range(n_iters):
        print("iter", i)
        overlap_found = False
        for i in range(len(text_objects)):
            for j in range(i + 1, len(text_objects)):
                # Get bounding boxes
                bbox_i = text_objects[i].get_window_extent(renderer=plt.gcf().canvas.get_renderer())
                bbox_j = text_objects[j].get_window_extent(renderer=plt.gcf().canvas.get_renderer())

                # Check for overlap
                intersection_area = max(0, min(bbox_i.x1, bbox_j.x1) - max(bbox_i.x0, bbox_j.x0)) * max(0, min(bbox_i.y1, bbox_j.y1) - max(bbox_i.y0, bbox_j.y0))
                union_area = (bbox_i.width * bbox_i.height) + (bbox_j.width * bbox_j.height) - intersection_area
                overlap_rate = intersection_area / union_area

                # If overlap exceeds threshold, adjust y-coordinate of one text label
                if overlap_rate > threshold:
                    pos = text_objects[i].get_position()
                    text_objects[i].set_text(str(i)) # Adjust y-coordinate
                    text_objects[i].set_position((pos[0], pos[1] + 0.01)) # Adjust y-coordinate
                    text_objects[i].set_color(colors[i]) # Adjust y-coordinate
                    legend_idxs.append(i)

    return set(list(legend_idxs))

def plot_text_embedding(X, colors, labels, fig_fp=None):
    pca = PCA(2)
    Comps = pca.fit_transform(X)

    # Function to calculate overlap rate
    def overlap_rate(bbox1, bbox2):
        intersection_area = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0)) * max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
        union_area = (bbox1.width * bbox1.height) + (bbox2.width * bbox2.height) - intersection_area
        return intersection_area / union_area

    fig = plt.figure(figsize=(20, 12))
    # fig, ax = plt.subplots()
    ax = fig.add_subplot()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    ax.scatter(Comps[:,0], Comps[:,1], c=colors) # , s= # TODO: change size
    text_objects = [ax.text(Comps[i,0], Comps[i,1], label, size=14) for i, label in enumerate(labels)] # , c=Y[i], ha='center', va='center'
    legend_idxs = check_and_adjust_overlap(text_objects, colors, 10)
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label="%3d "%i + label) for i, label in enumerate(labels) if i in legend_idxs]
    ax.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(0.8, 0.5))
    
    # legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label="%3d "%i + label) for i, label in enumerate(labels)]
    # ax.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # Turn off the outer boundary (spines) and axis
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])
    # plt.title("Class Embedding Distribution", fontsize=16, x=0.7, y=0.98)  

    if fig_fp is None:
        fig_fp = os.path.join(os.path.dirname(__file__), "temp", "TextEmbedding.png")

    plt.savefig(fig_fp)
    print("file saved to ", fig_fp)


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

    # X = dataset_train.text_features_vc.detach().cpu().numpy()
    X = np.load(os.path.join(os.path.dirname(__file__), "assets", "text_features.npy"))
    colors = df_action["color"]
    labels = df_action['action'].values
    plot_text_embedding(X, colors, labels)

  