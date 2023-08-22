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
        self.ckpt_path_ic = config['ckpt_path_ic']
            
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

    def produce_prompt_embedding(self, video_clip, force=False):
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        npy_fp_vc = os.path.join(temp_dir, "text_features_vc.npy")
        if not os.path.exists(npy_fp_vc) or force:
            prompts = self.df_action['prompt'].tolist()
            prompts = tokenize(prompts).to(self.device)       
            with torch.no_grad():
                text_features_vc = video_clip.encode_text(prompts)
                text_features_vc = torch.nn.functional.normalize(text_features_vc, dim=1)
            np.save(npy_fp_vc, text_features_vc.cpu().detach().numpy())
            self.text_features_vc = text_features_vc.float()
        else:
            self.text_features_vc = torch.from_numpy(np.load(npy_fp_vc)).to(self.device).float()

        npy_fp_ic = os.path.join(temp_dir, "text_features_ic.npy")
        if not os.path.exists(npy_fp_ic) or force:
            from open_clip import load_openai_model, get_tokenizer
            clip_name = os.path.basename(self.ckpt_path_ic).split(".")[0]
            clip_ic = load_openai_model(self.ckpt_path_ic).to(self.device)
            text = get_tokenizer(clip_name)(self.df_action['prompt']).to(self.device)
            with torch.no_grad():
                text_features_ic = clip_ic.encode_text(text)
                text_features_ic = torch.nn.functional.normalize(text_features_ic)
            np.save(npy_fp_ic, text_features_ic.cpu().detach().numpy())
            self.text_features_ic = text_features_ic.float()
        else:
            self.text_features_ic = torch.from_numpy(np.load(npy_fp_ic)).to(self.device).float()

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

        # fast stream
        self.enable_image_clip = config['enable_image_clip']
        self.enable_african = config['enable_african']

        at_least_one_source = self.enable_video_clip or self.enable_image_clip or self.enable_african
        assert at_least_one_source, "at least one data source should be enabled"

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1
        frames_tensor = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
        frames_tensor = self.video_aug(frames_tensor, self.video_transform)            
        return frames_tensor, labels_onehot, index
             
class AnimalKingdomDatasetVisualize(AnimalKingdomDataset):
    """two samplings of video for two streams of model"""
    def __init__(self, config, split=""):
        super().__init__(config, split)
        self.video_transform = VideoTransformTorch(mode='val')  # train or val model

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1
        video_raw = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
        video_aug = self.video_aug(video_raw, self.video_transform)            
        return video_fp, video_raw, video_aug, labels_onehot, index
                          
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

