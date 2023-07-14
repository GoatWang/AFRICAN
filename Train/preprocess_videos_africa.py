import os
import json
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch import utils
from config import config
from Model import VideoCLIP
from config import ex, config
from InternVideo import tokenize
from Dataset import AnimalKingdomDataset
from open_clip import _build_vision_tower

def load_clip_africa(config):
    africa_config_fp = os.path.join(os.path.dirname(__file__), "open_clip/model_configs/ViT-L-14.json")
    with open(africa_config_fp, 'r') as f:
        africa_model_config = json.load(f)
    clip_africa = _build_vision_tower(africa_model_config['embed_dim'], africa_model_config['vision_cfg'])
    if config['original_clip_africa']:
        # original_clip
        state_dict_africa = torch.jit.load(config['ckpt_path_africa'], map_location="cpu").visual.state_dict()
    else:
        # pretrained africa clip
        state_dict_africa = torch.load(config['ckpt_path_africa'], map_location="cpu")['state_dict']
        state_dict_africa = {name.replace("image_encoder.", ""): weights for name, weights in state_dict_africa.items() if "image_encoder" in name}
    clip_africa.load_state_dict(state_dict_africa)
    clip_africa.to(config['device'])
    clip_africa.requires_grad = False
    clip_africa.eval()
    return clip_africa

@ex.automain
def main(_config):
    dataset_train = AnimalKingdomDataset(_config, preprocessed=False, split="train")
    dataset_valid = AnimalKingdomDataset(_config, preprocessed=False, split="val")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = VideoCLIP(_config)
    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=False)
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False)

    clip_africa = load_clip_africa(_config)
    Path(_config['preprocess_dir']).mkdir(parents=True, exist_ok=True)
    
    for batch_idx, (video_feats_africa, video_fp) in enumerate(tqdm(train_loader)):
        video_feats_africa = video_feats_africa.to(_config['device'])
        video_feats_africa = clip_africa(video_feats_africa)
        torch.save(video_feats_africa, os.path.join(_config['preprocess_dir'], os.path.basename(video_fp).split(".")[0] + ".pt"))

    for batch_idx, (video_feats_africa, video_fp) in enumerate(tqdm(valid_loader)):
        video_feats_africa = video_feats_africa.to(_config['device'])
        video_feats_africa = clip_africa(video_feats_africa)
        torch.save(video_feats_africa, os.path.join(_config['preprocess_dir'], os.path.basename(video_fp).split(".")[0] + ".pt"))

