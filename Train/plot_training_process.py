import os
import pandas as pd
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def plot_progress(df, n_rows, title, xaxis, yaxis, xlabel, ylabel, fig_fp=None):
    assert type(yaxis) is list, "xaxis should be list type"

    df = df.loc[:n_rows]
    plt.figure(figsize=(7, 4))

    for y in yaxis:
        plt.plot(df.loc[:n_rows, xaxis], df.loc[:n_rows, y], label=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="lower right")

    if fig_fp is None:
        # os.path.join(os.path.dirname(__file__), "temp", "training_progress.png")
        os.path.join(os.path.dirname(__file__), "temp", "training_progress.pgf")
        
    plt.savefig(fig_fp)
    print("file saved to ", fig_fp)

if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_csv("assets/training_process/BackboneSelection.csv")
    save_dir = os.path.join(os.path.dirname(__file__), "temp", "training_progress")
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    title = 'mAP performance on each Epoch for VC_Vision, VC_Proj, and IC'
    fig_fp = os.path.join(save_dir, "BackboneSelection.pgf")
    plot_progress(df, n_rows=70, title=title, xaxis='epoch', yaxis=['VC_Vision', 'VC_Proj', 'IC'], xlabel='Epoch', ylabel='mAP', fig_fp=fig_fp)



