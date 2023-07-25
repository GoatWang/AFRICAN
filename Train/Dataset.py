import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from InternVideo import tokenize
from PromptEngineer import generate_prompt
from Transform import VideoTransformTorch, video_aug
from VideoReader import read_frames_decord, read_feats_fast

class AnimalKingdomDataset(torch.utils.data.Dataset):
    """single sampling of video for two streams of model"""
    def __init__(self, config, split=""):
        assert split in ["train", "val"], "split must be train or val"
        self.split = split
        self.metadata = None
        self.ans_lab_dict = dict()
        self.device = config['device'] 
        self.data_dir = config['data_dir']
        self.n_classes = config['n_classes']
        self.num_frames = config['num_frames']
        self.video_sampling = config['video_sampling']
        self.functional_test_size = config['functional_test_size']
            
        # self.text_column_name = "questions"
        self.video_transform = VideoTransformTorch(mode=self.split)  # train or val model
        self.video_aug = video_aug
        self.load_metadata()

    def process_annotation(self, csv_fp, video_fps):
        video_id_mapping = {os.path.basename(fp).replace(".mp4", ""):fp for fp in video_fps}

        # group into one video per line
        df = pd.read_csv(csv_fp, sep=' ')
        df_out1 = df.groupby("original_vido_id").first().reset_index()
        df_out2 = df.groupby("original_vido_id")['path'].apply(len).reset_index()
        df_out2.columns = ['original_vido_id', 'count']
        df_out = pd.merge(df_out1.drop('path', axis=1), df_out2, on='original_vido_id')

        # add features
        df_out['video_fp'] = df_out['original_vido_id'].apply(video_id_mapping.get)
        df_out['labels'] = df_out['labels'].apply(lambda x: [int(l) for l in x.split(",")])
        df_out = df_out[df_out['video_fp'].notnull()].reset_index(drop=True)
        end_idx = self.functional_test_size if self.functional_test_size else len(df_out)
        return df_out['video_fp'].loc[:end_idx].tolist(), df_out['labels'].loc[:end_idx].tolist()

    def load_metadata(self):
        self.df_action = pd.read_excel(os.path.join(self.data_dir, 'annotation', 'df_action.xlsx'))
        self.df_action['prompt'] = self.df_action.apply(generate_prompt, axis=1)
        # self.df_metadata = pd.read_excel(os.path.join(self.data_dir, 'AR_metadata.xlsx'))
        video_fps = glob.glob(os.path.join(self.data_dir, 'dataset', 'video', "*.mp4"))
        split_files = {
            'train': os.path.join(self.data_dir, "annotation", "train.csv"),
            'val': os.path.join(self.data_dir, "annotation", "val.csv")
        }
        target_split_fp = split_files[self.split]
        self.video_fps, self.labels = self.process_annotation(target_split_fp, video_fps)

    def produce_prompt_embedding(self, clipmodel, force=False):
        npy_fp = os.path.join("temp", "text_features.npy")
        if not os.path.exists(npy_fp) or force:
            prompts = self.df_action['prompt'].tolist()
            prompts = tokenize(prompts).to(self.device)       
            with torch.no_grad():
                text_features = clipmodel.encode_text(prompts)
                text_features = torch.nn.functional.normalize(text_features, dim=1)
            Path(os.path.dirname(npy_fp)).mkdir(parents=True, exist_ok=True)
            np.save(npy_fp, text_features.cpu().detach().numpy())
            self.text_features = text_features
        else:
            self.text_features = torch.from_numpy(np.load(npy_fp)).to(self.device)

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        frames_tensor = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
        frames_tensor = self.video_aug(frames_tensor, self.video_transform)            
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1

        return frames_tensor, labels_onehot, index
    
    def __len__(self):
        return len(self.video_fps)
    
class AnimalKingdomDatasetDiffSampling(AnimalKingdomDataset):
    """two samplings of video for two streams of model"""
    def __init__(self, config, split=""):
        super(self).__init__(config, split)
        self.preprocess_dir = config['preprocess_dir']
        self.ckpt_path_fast = config['ckpt_path_fast']
        self.use_image_clip_fast = config['use_image_clip_fast']
        self.num_frames_fast = config['num_frames_fast']
        self.video_sampling_fast = config['video_sampling_fast']
        self.enable_preprocess_fast = config['enable_preprocess_fast']
        self.video_transform_fast = VideoTransformTorch(mode='val')  # do not transform

    def get_preprocess_feats_fp(self, video_fp):
        base_model_name_fast = os.path.basename(self.ckpt_path_fast).split('.')[0]
        save_dir = os.path.join(self.preprocess_dir, base_model_name_fast)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(save_dir, os.path.basename(video_fp).split(".")[0] + ".pt")

    def __getitem__(self, index):
        video_fp = self.video_fps[index]

        # slow stream
        frames_tensor = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
        frames_tensor = self.video_aug(frames_tensor, self.video_transform)            
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1

        # fast stream
        self.enable_preprocess_fast
        if self.enable_preprocess_fast:
            video_feats_fast_fp = self.get_preprocess_feats_fp(video_fp)
            frames_feats_tensor_fast = read_feats_fast(video_feats_fast_fp, self.num_frames_fast, self.video_sampling_fast)
            return frames_tensor, frames_feats_tensor_fast, labels_onehot, index
        else:
            frames_tensor_fast = read_frames_decord(video_fp, num_frames=self.num_frames_fast, sample=self.video_sampling_fast)[0]
            frames_tensor_fast = self.video_aug(frames_tensor_fast, self.video_transform_fast)
            return frames_tensor, frames_tensor_fast, labels_onehot, index
             
class AnimalKingdomDatasetPreprocess(AnimalKingdomDatasetDiffSampling):
    def __init__(self, config, split=""):
        super(self).__init__(config, split)

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        video_feats_fast_fp = self.get_preprocess_feats_fp(video_fp)
        if not os.path.exists(video_feats_fast_fp):
            video_tensor_fast = read_frames_decord(video_fp, num_frames=self.num_frames_fast, sample='all')[0]
            video_tensor_fast = self.video_aug(video_tensor_fast, self.video_transform_fast)
        else:
            # FOR ACCERLATION
            video_tensor_fast = torch.zeros(1)

        return video_tensor_fast, video_fp


if __name__  == "__main__":
    from config import config
    _config = config()

    from torch import utils
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    print("len(train_loader)", len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
      video_tensor, video_feats_fast, labels_onehot, index = batch
      print("video_tensor.shape", video_tensor.shape)
      print("video_feats_fast.shape", video_feats_fast.shape)
      print("labels_onehot.shape", labels_onehot.shape)
      print("index", index)
      print(batch_idx, "success")
      break

    # print(len(valid_loader))
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
    #   print(batch_idx, "success")
    #   break

    dataset = AnimalKingdomDataset(_config, split="train")
    df_action = dataset.df_action
    video_tensor, video_feats_fast, labels_onehot, index = dataset[0]
    print(video_tensor.shape)
    print(labels_onehot.shape)
    for idx, prompt in df_action.loc[np.where(labels_onehot)[0], 'prompt'].items():
        print(str(idx).zfill(3) + ":", prompt)
    
    from Model import VideoCLIP
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = VideoCLIP(_config)
    dataset.produce_prompt_embedding(model.clip)
    print(dataset.text_features.shape)

