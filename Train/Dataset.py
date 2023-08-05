import os
import json
import glob
import torch
import random
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
    
class AnimalKingdomDatasetSlowFast(AnimalKingdomDataset):
    """two samplings of video for two streams of model"""
    def __init__(self, config, split=""):
        super().__init__(config, split)

        # slow stream
        self.enable_video_clip = config['enable_video_clip']

        # fast stream: image clip
        self.preprocess_dir = config['preprocess_dir']
        self.num_preaug_videos = config['num_preaug_videos']
        self.enable_image_clip = config['enable_image_clip']
        self.suffix_zfill_number = config['suffix_zfill_number']
        if self.enable_image_clip:
            self.ckpt_path_ic = config['ckpt_path_ic']
            self.num_frames_ic = config['num_frames_ic']
            self.video_sampling_ic = config['video_sampling_ic']
            self.enable_preprocess_ic = config['enable_preprocess_ic']
            self.video_transform_ic = VideoTransformTorch(mode=split)  # do not transform

        # fast stream: african
        self.enable_african = config['enable_african']
        if self.enable_african:
            self.ckpt_path_af = config['ckpt_path_af']
            self.num_frames_af = config['num_frames_af']
            self.video_sampling_af = config['video_sampling_af']
            self.enable_preprocess_af = config['enable_preprocess_af']
            self.video_transform_af = VideoTransformTorch(mode=split)  # do not transform

        at_least_one_source = self.enable_video_clip or self.enable_image_clip or self.enable_african
        assert at_least_one_source, "at least one data source should be enabled"

    def get_preprocess_feats_fp(self, video_fp, pretrained_type, suffix='_00'):
        """
        pretrained_type: {"ic", "af"}
        """
        if pretrained_type == "ic":
            ckpt_path = self.ckpt_path_ic  
        elif pretrained_type == "af":
            ckpt_path = self.ckpt_path_af
        else:
            raise NotImplementedError
        base_model_name_fast = os.path.basename(ckpt_path).split('.')[0]
        save_dir = os.path.join(self.preprocess_dir, base_model_name_fast)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(save_dir, os.path.basename(video_fp).split(".")[0] + suffix + ".pt")

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1

        # slow stream
        frames_tensor_vc = torch.zeros(1, 3, 224, 224)
        if self.enable_video_clip:
            frames_tensor_vc = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
            frames_tensor_vc = self.video_aug(frames_tensor_vc, self.video_transform)            

        # fast stream: image clip
        frames_tensor_ic = torch.zeros(1, 3, 224, 224)
        if self.enable_image_clip:
            if self.enable_preprocess_ic:
                suffix = "_"+str(random.randint(self.num_preaug_videos)).zfill(self.suffix_zfill_number)
                video_feats_fast_fp = self.get_preprocess_feats_fp(video_fp, "ic", suffix=suffix)
                frames_tensor_ic = torch.load(video_feats_fast_fp)
            else:
                frames_tensor_ic = read_frames_decord(video_fp, num_frames=self.num_frames_ic, sample=self.video_sampling_ic)[0]
                frames_tensor_ic = self.video_aug(frames_tensor_ic, self.video_transform_ic)

        # fast stream: african
        frames_tensor_af = torch.zeros(1, 3, 224, 224)
        if self.enable_african:
            if self.enable_preprocess_af:
                video_feats_fast_fp = self.get_preprocess_feats_fp(video_fp, "ic", suffix="_"+str(random.randint(0, 30)).zfill(2))
                frames_tensor_ic = torch.load(video_feats_fast_fp)
            else:
                frames_tensor_af = read_frames_decord(video_fp, num_frames=self.num_frames_af, sample=self.video_sampling_af)[0]
                frames_tensor_af = self.video_aug(frames_tensor_af, self.video_transform_af)

        return frames_tensor_vc, frames_tensor_ic, frames_tensor_af, labels_onehot, index
             
class AnimalKingdomDatasetPreprocess(AnimalKingdomDatasetSlowFast):
    def __init__(self, config, pretrained_type, split=""):
        super().__init__(config, split)
        self.pretrained_type = pretrained_type
        
        if pretrained_type == "ic":
            self.num_frames = self.num_frames_ic
            self.video_sampling = self.video_sampling_ic
            self.video_transform = self.video_transform_ic # depends on split
        elif pretrained_type == "af":
            self.num_frames = self.num_frames_af
            self.video_sampling = self.video_sampling_af
            self.video_transform = self.video_transform_af # depends on split
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        suffixes = ["_" + str(i).zfill(self.suffix_zfill_number) for i in range(self.num_preaug_videos)]
        video_feats_fast_fps = [os.path.exists(self.get_preprocess_feats_fp(video_fp, self.pretrained_type, suffix)) for suffix in suffixes]
        if not np.all(video_feats_fast_fps):
            video_tensor_fast_aug = []
            for i in range(self.num_preaug_videos):
                video_tensor_fast_aug.append(read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]) # 0.1s
            video_tensor_fast_aug = torch.stack(video_tensor_fast_aug)
        else:
            # FOR ACCERLATION
            video_tensor_fast_aug = torch.zeros(self.num_preaug_videos, 1, 3, 224, 224)

        # video_tensor_fast.shape = self.num_preaug_videos, num_frames, 3, 224, 224
        return video_tensor_fast_aug, video_fp


if __name__  == "__main__":
    from torch import utils
    from config import config
    _config = config()

    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=_config['data_workers'])
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=_config['data_workers'])
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
    dataset.produce_prompt_embedding(model.video_clip)
    print(dataset.text_features.shape)

