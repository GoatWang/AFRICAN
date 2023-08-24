import os
import glob
import yaml
import wandb
import numpy as np
import pandas as pd
from pathlib import Path

target_dir = "/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/experiments/VCLs1at_B_nodecay_016_00015_008"
csv_fps = sorted(glob.glob(os.path.join(target_dir, "*", "*.csv")))
df = pd.concat([pd.read_csv(csv_fp) for csv_fp in csv_fps])

# # write to wandb
yaml_fp = "/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/experiments/VCLs1at_B_nodecay_016_00015_008/20230824-052908/hparams.yaml"
with open(yaml_fp, 'r') as f:
    config = yaml.safe_load(f.read())
with open("wandb_key.txt", 'r') as f:
    key = f.read()
wandb.login(key=key)
wandb.init(project='AnimalKingdom', name='VCLs1at_B_nodecay_016_00015_008', id='20230814-040728', config=config)
from pprint import pprint
for idx, row in df.iterrows():
    row_dict = row[row.notna()].to_dict()
    # pprint(row_dict)
    wandb.log(row_dict)
