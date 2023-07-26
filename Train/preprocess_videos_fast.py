import os
import copy
import json
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch import utils
from config import config
from config import ex, config
from InternVideo import tokenize
from Model import AfricanSlowfast
from open_clip import _build_vision_tower
from Dataset import AnimalKingdomDatasetPreprocess

def collate_func(batch):
    video_tensors = [item[0] for item in batch]
    video_tensors_n_frames = [video_tensor_fast.shape[0] for video_tensor_fast in video_tensors]
    video_tensors_cat = torch.cat(video_tensors, dim=0)
    video_fps = [item[1] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    return video_tensors_cat, video_tensors_n_frames, video_fps

def inference_save(_config, dataset, dataloader, image_encoder):
    with torch.no_grad():
        for batch_idx, (batch) in enumerate(tqdm(dataloader)):
            video_tensors_cat, video_tensors_n_frames, video_fps = batch
            video_tensors_fast = video_tensors_fast.to(_config['device'])
            for idx, (video_tensor_fast, video_fp) in enumerate(zip(video_tensors_fast, video_fps)):
                video_feats_fast_fp = dataset.get_preprocess_feats_fp(video_fp)
                if not os.path.exists(video_feats_fast_fp):
                    video_feats_fast = image_encoder(video_tensor_fast)
                    torch.save(video_feats_fast, video_feats_fast_fp)

def batch_inference_save(_config, dataset, dataloader, image_encoder):
    with torch.no_grad():
        for batch_idx, (batch) in enumerate(tqdm(dataloader)):
            video_tensors_cat, video_tensors_n_frames, video_fps = batch
            all_exists = np.all([os.path.exists(dataset.get_preprocess_feats_fp(video_fp)) for video_fp in video_fps])
            if all_exists:
                continue

            # parallel inference
            video_tensors_cat = video_tensors_cat.to(_config['device'])
            video_feats_tensors_cat = torch.zeros(video_tensors_cat.shape[0], _config['transformer_width_fast'])
            n_iters = (video_feats_tensors_cat.shape[0] // _config['preprocess_batch_size']) + 1
            for idx in range(n_iters):
                st, end = idx*_config['preprocess_batch_size'], (idx+1)*_config['preprocess_batch_size']
                video_tensors_batch = video_feats_tensors_cat[st:end]
                video_feats_tensors_batch = image_encoder(video_tensors_batch)
                video_feats_tensors_cat[st:end] = video_feats_tensors_batch

            # split into different videos $ save
            frane_st = 0
            for idx, (n_frames, video_fp) in enumerate(zip(video_tensors_n_frames, video_fps)):
                if not os.path.exists(video_feats_fast_fp):
                    video_feats_fast = video_feats_tensors_cat[frane_st: frane_st+n_frames]
                    frane_st = frane_st+n_frames
                    
                    video_feats_fast_fp = dataset.get_preprocess_feats_fp(video_fp)
                    torch.save(video_feats_fast, video_feats_fast_fp)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    dataset_train = AnimalKingdomDatasetPreprocess(_config, split="train")
    dataset_valid = AnimalKingdomDatasetPreprocess(_config, split="val")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = AfricanSlowfast(_config)
    dataset_train.produce_prompt_embedding(model.video_clip)
    dataset_valid.produce_prompt_embedding(model.video_clip)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], collate_func=collate_func, shuffle=False) # _config['batch_size']
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], collate_func=collate_func, shuffle=False) # _config['batch_size']

    image_encoder_fast = model.get_image_encoder_fast(_config)
    image_encoder_fast.to(_config['device'])
    image_encoder_fast.eval()

    batch_inference_save(_config, dataset_train, train_loader, image_encoder_fast)
    batch_inference_save(_config, dataset_valid, valid_loader, image_encoder_fast)

    # inference_save(_config, dataset_train, train_loader, image_encoder_fast)
    # inference_save(_config, dataset_valid, valid_loader, image_encoder_fast)
