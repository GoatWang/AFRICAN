import os
import glob
import yaml
import wandb
import numpy as np
import pandas as pd
from pathlib import Path

# load data
action_fp = "/Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/annotation/df_action.xlsx"
df_action = pd.read_excel(action_fp)

train_fp = "/Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/annotation/train.csv"
df_train = pd.read_csv(train_fp, sep=' ')
train_count = (np.bincount(np.hstack(df_train['labels'].apply(lambda x: [int(l) for l in x.split(",")]).tolist())))

valid_fp = "/Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/annotation/val.csv"
df_valid = pd.read_csv(valid_fp, sep=' ')
valid_count = (np.bincount(np.hstack(df_valid['labels'].apply(lambda x: [int(l) for l in x.split(",")]).tolist())))

# for i in range(140):
#     print(df_action['action'].loc[i], df_action['count'].loc[i], train_count[i], valid_count[i])

# simple statistics
df_action['train_count'] =  train_count
df_action['valid_count'] =  valid_count
df_actions_head = df_action.loc[df_action['segment'] == 'head', ['action', 'train_count', 'valid_count']]
df_actions_middle = df_action.loc[df_action['segment'] == 'middle', ['action', 'train_count', 'valid_count']]
df_actions_tail = df_action.loc[df_action['segment'] == 'tail', ['action', 'train_count', 'valid_count']]


# row content modification (MultilabelAccuracy, MultilabelExactMatch, MultilabelAveragePrecision, map_head, map_middle, map_tail, lr-AdamW)
target_dir = "/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/experiments/1_20230615-190925_414_033"
csv_fps = sorted(glob.glob(os.path.join(target_dir, "*", "*.csv")))
df = pd.concat([pd.read_csv(csv_fp) for csv_fp in csv_fps])
rows_dst = []
for idx, row in df.iterrows():
    row_dict = row.to_dict()
    if  pd.notna(row_dict['MultilabelAccuracy']) and pd.notna(row_dict['valid_loss']):
        for idx, col in enumerate(list(row_dict.keys())):
            if "Multilabel" in col:
                row_dict["valid_" + col] = row_dict[col]
                row_dict.pop(col)
    elif pd.notna(row_dict['MultilabelAccuracy']) and pd.isna(row_dict['valid_loss']):
        for idx, col in enumerate(list(row_dict.keys())):
            if "Multilabel" in col:
                row_dict["train_" + col] = row_dict[col]
                row_dict.pop(col)
    else:
        for idx, col in enumerate(list(row_dict.keys())):
            if "Multilabel" in col:
                row_dict["train_" + col] = np.nan
                row_dict["valid_" + col] = np.nan
                row_dict.pop(col)

    # row_dict['valid_map_head'] = np.nan
    # row_dict['valid_map_middle'] = np.nan
    # row_dict['valid_map_tail'] = np.nan
    # if pd.notna(row_dict['valid_map_Abseiling']):
    #     series = pd.Series(row_dict)
    #     series.index = [i.replace("valid_map_", "") for i in series.index]
    #     row_dict['valid_map_head'] = sum(series[df_actions_head['action']].values * df_actions_head['valid_count'].values) / sum(df_actions_head['valid_count'].values)
    #     row_dict['valid_map_middle'] = sum(series[df_actions_middle['action']].values * df_actions_middle['valid_count'].values) / sum(df_actions_middle['valid_count'].values)
    #     row_dict['valid_map_tail'] = sum(series[df_actions_tail['action']].values * df_actions_tail['valid_count'].values) / sum(df_actions_tail['valid_count'].values)

    # row_dict['train_map_head'] = np.nan
    # row_dict['train_map_middle'] = np.nan
    # row_dict['train_map_tail'] = np.nan        
    # if pd.notna(row_dict['train_map_Abseiling']):
    #     series = pd.Series(row_dict)
    #     series.index = [i.replace("train_map_", "") for i in series.index]
    #     row_dict['train_map_head'] = sum(series[df_actions_head['action']].values * df_actions_head['train_count'].values) / sum(df_actions_head['train_count'].values)
    #     row_dict['train_map_middle'] = sum(series[df_actions_middle['action']].values * df_actions_middle['train_count'].values) / sum(df_actions_middle['train_count'].values)
    #     row_dict['train_map_tail'] = sum(series[df_actions_tail['action']].values * df_actions_tail['train_count'].values) / sum(df_actions_tail['train_count'].values)

    row_dict['lr-AdamW'] = np.nan
    if pd.isna(row_dict["train_loss"]) and pd.isna(row_dict["epoch"]) and pd.isna(row_dict["valid_loss"]):
        row_dict['lr-AdamW'] = 0.00015
    rows_dst.append(row_dict)

fp = "/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/experiments/2_CLIP_BCE_nodecay_008/20230627-105353/metrics.csv"
drop_cols = ['valid_map_head', 'valid_map_middle', 'valid_map_tail', 'train_map_head', 'train_map_middle', 'train_map_tail']
target_columns = [c for c in list(pd.read_csv(fp).columns) if c not in drop_cols]
df = pd.DataFrame(rows_dst)[target_columns]

# save to csv
Path('temp').mkdir(parents=True, exist_ok=True)
df.to_csv(os.path.join('temp', "insert_wandb_metrics.csv"), index=False)


# write to wandb
yaml_fp = "/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/experiments/1_20230615-190925_414_033/20230616-100856/hparams.yaml"
with open(yaml_fp, 'r') as f:
    config = yaml.safe_load(f.read())
wandb.login(key='427974ce1dec11546ede262db4206a90fcf9ce00')
wandb.init(project='AnimalKingdom', name='CLIP_BCE_nodecay_128_00015', config=config)
for idx, row in df.iterrows():
    row_dict = row[row.notna()].to_dict()
    wandb.log(row_dict)
