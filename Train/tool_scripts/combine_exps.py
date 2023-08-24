import os
import glob
import pandas as pd
from matplotlib import pyplot as plt

base_dir = os.path.dirname(__file__)
exp_dirs = [os.path.join(base_dir, "..", "experiments", "VCLs1at_B_nodecay_016_00015_008")]
# for d in os.listdir(experiment_dir): 
#     target_dir = os.path.join(experiment_dir, d)
#     if os.path.isdir(target_dir) and (not d.startswith(".")):
for exp_dir in exp_dirs: 
    csv_fps_src = glob.glob(os.path.join(exp_dir, "*", "*.csv"), recursive=True)
    csv_fp_dst = os.path.join(exp_dir, os.path.basename(exp_dir)+'.csv')
    # print(csv_fps_src)
    df = pd.concat([pd.read_csv(csv_fp) for csv_fp in csv_fps_src])
    # print(df.head())
    columns = ['epoch', 'step', 'valid_loss', 'valid_MultilabelAccuracy', 'valid_MultilabelExactMatch', 'valid_MultilabelAveragePrecision']
    df_valid = df.loc[df['valid_loss'].notna(), columns].sort_values('step')
    df_valid.to_csv(csv_fp_dst, index=False)


# df_compare = None
# # for d in os.listdir(experiment_dir): 
# #     target_dir = os.path.join(experiment_dir, d)
# #     if os.path.isdir(target_dir) and (not d.startswith(".")):
# for exp_dir in exp_dirs: 
#     csv_fp_src = os.path.join(target_dir, d+'.csv')

#     df_temp = pd.read_csv(csv_fp_src)[['epoch', 'valid_MultilabelAveragePrecision']]
#     df_temp.columns = ['epoch', d]
#     if df_compare is None:
#         df_compare = df_temp
#     else:
#         df_compare = pd.merge(df_compare, df_temp, on='epoch', how='outer')

# # df_compare.index = df_compare['epoch']
# df_compare.plot(x='epoch', y=df_compare.columns[1:], figsize=(20, 10))
# plt.savefig(os.path.join(experiment_dir, "compare.png"))
# df_compare.to_csv(os.path.join(experiment_dir, "compare.csv"), index=False)



