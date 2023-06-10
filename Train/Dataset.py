import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from InternVideo import tokenize
from PromptEngineer import generate_prompt
from VideoReader import read_frames_decord
from Transform import VideoTransformTorch, video_aug

class AnimalKingdomDataset(torch.utils.data.Dataset):
    def __init__(self, config, split=""):
        assert split in ["train", "val"], "split must be train or val"
        self.split = split
        self.metadata = None
        self.ans_lab_dict = dict()
        self.data_dir = config['data_dir']
        self.n_classes = config['n_classes']

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
        return df_out['video_fp'].tolist(), df_out['labels'].tolist()

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
            prompts = tokenize(prompts)#.cuda()        
            with torch.no_grad():
                text_features = clipmodel.encode_text(prompts)
                text_features = torch.nn.functional.normalize(text_features, dim=1)
            Path(os.path.dirname(npy_fp)).mkdir(parents=True, exist_ok=True)
            np.save(npy_fp, text_features)
            self.text_features = text_features
        else:
            self.text_features = np.load(npy_fp)

    def __getitem__(self, index):
        ret = None
        video_fp = self.video_fps[index]
        video_tensor = read_frames_decord(video_fp)[0]
        video_tensor = self.video_aug(video_tensor, self.video_transform)
        labels_onehot = torch.zeros(self.n_classes)
        labels_onehot[self.labels[index]] = 1
        return video_tensor, labels_onehot
        # ret = {
        #     "video_idx": index,
        #     "video_fp": video_fp,
        #     "video": video_tensor,
        #     'labels': self.labels[index], # caption_idxs
        # }
        # return ret
    
    def __len__(self):
        return len(self.video_fps)

if __name__  == "__main__":
    from config import config
    from PromptEngineer import generate_prompt
    _config = config()
    dataset = AnimalKingdomDataset(_config, split="train")
    df_action = dataset.df_action
    video_tensor, labels_onehot = dataset[0]
    print(video_tensor.shape)
    print(labels_onehot.shape)
    for idx, prompt in df_action.loc[np.where(labels_onehot)[0], 'prompt'].items():
        print(str(idx).zfill(3) + ":", prompt)
    
    from Model import VideoCLIP
    model = VideoCLIP(_config)
    dataset.produce_prompt_embedding(model.clip)
    print(dataset.text_features.shape)

