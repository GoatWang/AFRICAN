import os
import glob
import pandas as pd

dirs = [d for d in os.listdir() if os.path.isdir(d)]
for d in dirs[:]:
    csv_fps_src = glob.glob(d + "/*/*.csv")
    csv_fp_dst = os.path.join(d, d+'.csv')
    df = pd.concat([pd.read_csv(csv_fp) for csv_fp in csv_fps_src])
    # columns = ['epoch', 'step', 'valid_loss', 'MultilabelAccuracy', 'MultilabelExactMatch', 'MultilabelAveragePrecision']
    # df_valid = df.loc[df['valid_loss'].notna(), columns].sort_values('step')
    df_valid = df.loc[df['valid_loss'].notna()].sort_values('step')
    df_valid.to_csv(csv_fp_dst, index=False)


