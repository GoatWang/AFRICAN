import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from open_clip import tokenize
from PromptEngineer import generate_prompt
from Transform import VideoTransformTorch, video_aug
from VideoReader import read_frames_decord, read_feats

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
        self.enable_preprocess = config['enable_preprocess']
        self.functional_test_size = config['functional_test_size']
            
        self.preprocess_dir = config['preprocess_dir']
        self.IC_ckpt_path = config['IC_ckpt_path']
        self.AF_ckpt_path = config['AF_ckpt_path']

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

    def get_preprocess_feats_fp(self, video_fp, stream="IC"):
        if stream == "IC":
            ckpt_path = self.IC_ckpt_path
        elif stream == "AF":
            ckpt_path = self.AF_ckpt_path
        else:
            raise NotImplementedError
        
        base_model_name = os.path.basename(ckpt_path).split('.')[0]
        save_dir = os.path.join(self.preprocess_dir, base_model_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(save_dir, os.path.basename(video_fp).split(".")[0] + ".pt")

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        labels_onehot = torch.zeros(self.n_classes, dtype=torch.int32)
        labels_onehot[self.labels[index]] = 1

        if self.enable_preprocess:
            video_feats_fp_ic = self.get_preprocess_feats_fp(video_fp, "IC")
            feats_tensor_ic, frame_idxs = read_feats(video_feats_fp_ic, self.num_frames, self.video_sampling)
            video_feats_fp_af = self.get_preprocess_feats_fp(video_fp, "AF")
            feats_tensor_af, frame_idxs = read_feats(video_feats_fp_af, self.num_frames, self.video_sampling, frame_idxs)
            return feats_tensor_ic, feats_tensor_af, labels_onehot, index
        else:
            frames_tensor = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)[0]
            frames_tensor = self.video_aug(frames_tensor, self.video_transform)
            return frames_tensor, labels_onehot, index
    
    def __len__(self):
        return len(self.video_fps)
    
class AnimalKingdomDatasetPreprocess(AnimalKingdomDataset):
    def __init__(self, config, split=""):
        super().__init__(config, split)
        self.video_transform = VideoTransformTorch(mode='val')  # train or val model

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        video_feats_fp = self.get_preprocess_feats_fp(video_fp)
        if not os.path.exists(video_feats_fp):
            video_tensor = read_frames_decord(video_fp, num_frames=self.num_frames, sample='all')[0]
            video_tensor = self.video_aug(video_tensor, self.video_transform)
        else:
            # FOR ACCERLATION
            video_tensor = torch.zeros(1, 3, 224, 224)

        return video_tensor, video_fp


# if __name__  == "__main__":
    # from config import config
    # _config = config()

    # from torch import utils
    # dataset_train = AnimalKingdomDataset(_config, split="train")
    # dataset_valid = AnimalKingdomDataset(_config, split="val")
    # train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    # valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    # print("len(train_loader)", len(train_loader))
    # for batch_idx, batch in enumerate(train_loader):
    #   video_tensor, video_feats_fast, labels_onehot, index = batch
    #   print("video_tensor.shape", video_tensor.shape)
    #   print("video_feats_fast.shape", video_feats_fast.shape)
    #   print("labels_onehot.shape", labels_onehot.shape)
    #   print("index", index)
    #   print(batch_idx, "success")
    #   break

    # # print(len(valid_loader))
    # # for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
    # #   print(batch_idx, "success")
    # #   break

    # dataset = AnimalKingdomDataset(_config, split="train")
    # df_action = dataset.df_action
    # video_tensor, video_feats_fast, labels_onehot, index = dataset[0]
    # print(video_tensor.shape)
    # print(labels_onehot.shape)
    # for idx, prompt in df_action.loc[np.where(labels_onehot)[0], 'prompt'].items():
    #     print(str(idx).zfill(3) + ":", prompt)
    
    # from Model import VideoCLIP
    # _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    # model = VideoCLIP(_config)
    # dataset.produce_prompt_embedding(model.video_clip)
    # print(dataset.text_features.shape)

