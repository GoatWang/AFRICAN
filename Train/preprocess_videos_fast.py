import os
import copy
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
from Model import AfricanSlowfast
from open_clip import _build_vision_tower
from Dataset import AnimalKingdomDatasetPreprocess

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    dataset_train = AnimalKingdomDatasetPreprocess(_config, preprocessing=True, split="train")
    dataset_valid = AnimalKingdomDatasetPreprocess(_config, preprocessing=True, split="val")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = AfricanSlowfast(_config)
    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # _config['batch_size']
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False) # _config['batch_size']

    image_encoder_fast = model.get_image_encoder_fast(_config)
    image_encoder_fast.to(config['device'])
    image_encoder_fast.eval()

    with torch.no_grad():
        for batch_idx, (video_tensors_fast, video_fps) in enumerate(tqdm(train_loader)):
            video_tensors_fast = video_tensors_fast.to(_config['device'])
            for idx, (video_tensor_fast, video_fp) in enumerate(zip(video_tensors_fast, video_fps)):
                video_feats_fast_fp = dataset_train.get_preprocess_feats_fp(video_fp)
                if not os.path.exists(video_feats_fast_fp):
                    video_feats_fast = image_encoder_fast(video_tensor_fast)
                    torch.save(video_feats_fast, video_feats_fast_fp)

        for batch_idx, (video_tensors_fast, video_fps) in enumerate(tqdm(valid_loader)):
            video_tensors_fast = video_tensors_fast.to(_config['device'])
            for idx, (video_tensor_fast, video_fp) in enumerate(zip(video_tensors_fast, video_fps)):
                video_feats_fast_fp = dataset_valid.get_preprocess_feats_fp(video_fp)
                if not os.path.exists(video_feats_fast_fp):
                    video_feats_fast = image_encoder_fast(video_tensor_fast)            
                    torch.save(video_feats_fast, video_feats_fast_fp)



