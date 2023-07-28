import os
import json
import math
from copy import deepcopy

import torch
from torch import nn
import torchmetrics
import numpy as np
from typing import Callable
from Loss import get_loss_func
import pytorch_lightning as pl
from open_clip import Transformer, LayerNorm, load_openai_model
from ModelUtil.clip_param_keys import clip_param_keys
from open_clip import _build_vision_tower
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

class FramesTransformer(nn.Module):
    def __init__(
            self,
            seq_len: int = 32,
            width: int = 768,
            layers: int = 12,
            heads: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        # class embeddings and positional embeddings
        scale = width ** -0.5
        
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(seq_len, width))

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, x: torch.Tensor):
        # class embeddings and positional embeddings
        # x = torch.cat(
        #     [
        #         self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #         x
        #     ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        # pooled, tokens = x[:, 0], x[:, 1:] # use cls token to do classification. # https://chat.openai.com/share/83c1c340-3a53-4366-901f-183fbc05f2a2
        pooled = x.mean(dim=1) # mean over all tokens to do classification. # https://chat.openai.com/share/83c1c340-3a53-4366-901f-183fbc05f2a2
        pooled = pooled @ self.proj
        return pooled


class AfricanClip(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["lr"]
        self.num_frames = config['num_frames']
        self.n_classes = config["n_classes"]
        self.optimizer = config["optimizer"]

        self.decay_power = config['decay_power']
        self.warmup_steps = config['warmup_steps']
        self.max_steps = config['max_steps']
        self.end_lr = config['end_lr']
        self.poly_decay_power = config['poly_decay_power']

        self.enable_african = config['enable_african']
        self.enable_preprocess = config['enable_preprocess']

        self.IC_ckpt_path = config['IC_ckpt_path']
        self.AF_ckpt_path = config['AF_ckpt_path']

        # load clip
        self.image_clip = load_openai_model(self.IC_ckpt_path)
        
        # image clip stream
        if not self.enable_preprocess:
            self.image_encoder_ic = self.image_clip.visual # self.get_image_encoder("IC")
            self.freeze_image_encoder_evl(self.image_encoder_ic)
        self.transformer_ic = FramesTransformer(
            self.num_frames,
            config['IC_transformer_width'],
            config['IC_transformer_layers'],
            config['IC_transformer_heads']
        )
        self.logit_scale_ic = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.final_fc_ic = torch.nn.Linear(self.n_classes, self.n_classes)

        # african stream
        if self.enable_african:
            if not self.enable_preprocess:
                self.image_encoder_af = self.get_image_encoder("AF")
                self.freeze_image_encoder_evl(self.image_encoder_af)

            self.transformer_af = FramesTransformer(
                self.num_frames,
                config['AF_transformer_width'],
                config['AF_transformer_layers'],
                config['AF_transformer_heads']
            )

            output_width = int(config['AF_transformer_width'])
            self.mlp_af = nn.Sequential(
                nn.Linear(output_width, output_width//2),
                nn.Linear(output_width//2, output_width//4),
                nn.Linear(output_width//4, self.n_classes)
            )

            self.w_ic = nn.Parameter(torch.ones(self.n_classes) * 0.9)
            self.w_af = nn.Parameter(torch.ones(self.n_classes) * 0.1)
            self.bias = nn.Parameter(torch.randn(self.n_classes))
            # self.final_fc_af = torch.nn.Linear(int(self.n_classes * 2), self.n_classes)

    def print_requires_grad(self, model):
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

    def get_image_encoder(self, stream='IC'):
        """
        stream: {"IC", "AF"} represens Image Clip and African
        """
        clip_model_config = { # "open_clip/model_configs/ViT-L-14.json"
            "embed_dim": 768,
            "vision_cfg": {
                "image_size": 224,
                "layers": 24,
                "width": 1024,
                "patch_size": 14
            },
        }
        image_encoder = _build_vision_tower(clip_model_config['embed_dim'], clip_model_config['vision_cfg'])
        if stream == "IC":
            # original_clip
            state_dict_clip = torch.jit.load(self.IC_ckpt_path, map_location="cpu").visual.state_dict()
            image_encoder.load_state_dict(state_dict_clip)
        elif stream == "AF":
            # pretrained african
            state_dict_african = torch.load(self.AF_ckpt_path, map_location="cpu")['state_dict']
            state_dict_african = {name.replace("image_encoder.", ""): weights for name, weights in state_dict_african.items() if "image_encoder" in name}
            image_encoder.load_state_dict(state_dict_african)
        else:
            raise NotImplementedError
        return image_encoder

    def load_ckpt_state_dict(self, ckpt_fp):
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def set_text_feats(self, text_feats):
        self.text_feats = text_feats.clone().requires_grad_(False)

    def set_class_names(self, class_names):
        self.class_names = class_names

    def set_loss_func(self, loss_name, class_frequency):
        class_frequency = list(map(float, class_frequency))
        self.loss_func = get_loss_func(loss_name, class_frequency, self.device)

    def set_metrics(self, classes_head, classes_middle, classes_tail):
        metric_collection = torchmetrics.MetricCollection([
            torchmetrics.classification.MultilabelAccuracy(num_labels=self.n_classes),
            torchmetrics.ExactMatch(task='multilabel', num_labels=self.n_classes),
            torchmetrics.classification.MultilabelAveragePrecision(num_labels=self.n_classes), # total
        ])
        self.train_metrics = metric_collection.clone(prefix='train_')
        self.valid_metrics = metric_collection.clone(prefix='valid_')

        self.classes_head = classes_head
        self.classes_middle = classes_middle
        self.classes_tail = classes_tail
        
        self.train_map_head = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_head)) # head
        self.train_map_middle = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_middle)) # middle
        self.train_map_tail = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_tail)) # tail

        self.valid_map_head = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_head)) # head
        self.valid_map_middle = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_middle)) # middle
        self.valid_map_tail = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(classes_tail)) # tail

        self.train_map_class = torchmetrics.classification.MultilabelAccuracy(num_labels=self.n_classes, average=None)
        self.valid_map_class = torchmetrics.classification.MultilabelAccuracy(num_labels=self.n_classes, average=None)

    def freeze_image_encoder_evl(self, model):
        for n, p in model.named_parameters():
            if (
                "visual" in n
                and "ln_post" not in n # and "visual_ln_post" not in n
                and "proj" not in n # and "visual_proj" not in n
            ):
                p.requires_grad = False
            elif "transformer" in n:
                p.requires_grad = False
            elif "token_embedding" in n:
                p.requires_grad = False
            elif "positional_embedding" in n:
                p.requires_grad = False
            elif "conv1" in n:
                p.requires_grad = False
            elif "ln_pre" in n:
                p.requires_grad = False
                
    def forward_frames_ic(self, frames_tensor):
        """encode image into embedding"""
        B, F, C, H, W = frames_tensor.shape
        frames_tensor = frames_tensor.view(B*F, C, H, W)
        frames_feats = self.image_encoder_ic(frames_tensor)
        frames_feats = frames_feats.view(B, F, -1)
        frames_feats = self.forward_frames_feats_ic(self, frames_feats)
        return frames_feats

    def forward_frames_feats_ic(self, frames_feats):
        """apply transformer on image embedding of each frames"""
        frames_feats = self.transformer_ic(frames_feats)
        return frames_feats
    
    def forward_frames_af(self, frames_tensor):
        """encode image into embedding"""
        B, F, C, H, W = frames_tensor.shape
        frames_tensor = frames_tensor.view(B*F, C, H, W)
        frames_feats = self.image_encoder_af(frames_tensor)
        frames_feats = frames_feats.view(B, F, -1)
        frames_feats = self.forward_frames_feats_af(self, frames_feats)
        return frames_feats

    def forward_frames_feats_af(self, frames_feats):
        """apply transformer on image embedding of each frames"""
        frames_feats = self.transformer_af(frames_feats)
        return frames_feats
    
    def encode_video_feats(self, batch):
        feats_tensor_ic, feats_tensor_af, labels_onehot, index = batch
        frames_feats_ic = self.forward_frames_feats_ic(feats_tensor_ic)

        frames_feats_af = None
        if self.enable_african:
            frames_feats_af = self.forward_frames_feats_af(feats_tensor_af)
        return frames_feats_ic, frames_feats_af, labels_onehot

    def encode_video_frames(self, batch):
        frames_tensor, labels_onehot, index = batch
        frames_feats_ic = self.forward_frames_ic(frames_tensor)

        frames_feats_af = None
        if self.enable_african:
            frames_feats_af = self.forward_frames_af(frames_tensor)
        return frames_feats_ic, frames_feats_af, labels_onehot

    def forward(self, batch):
        if self.enable_preprocess:
            frames_feats_ic, frames_feats_af, labels_onehot = self.encode_video_feats(batch)
        else:
            frames_feats_ic, frames_feats_af, labels_onehot = self.encode_video_frames(batch)

        # similarity calculation on image clip stream
        video_feats_ic = torch.nn.functional.normalize(frames_feats_ic, dim=1) # (n, 768)
        text_feats = torch.nn.functional.normalize(self.text_feats, dim=1) # (140, 768)
        t = self.logit_scale_ic.exp()
        video_logits = ((video_feats_ic @ text_feats.t()) * t) # (n, 140)
        video_logits = self.final_fc_ic(video_logits)
    
        if self.enable_african:
            video_logits_af = self.mlp_af(frames_feats_af)
            video_logits = self.w_ic * video_logits + self.w_af * video_logits_af + self.bias
            # video_logits = torch.cat([video_logits, video_logits_af], dim=1)
            # video_logits = self.final_fc_af(video_logits)

        return video_logits, labels_onehot
    
    def training_step(self, batch, batch_idx):
        video_logits, labels_onehot = self(batch)
        video_pred = torch.sigmoid(video_logits)
        loss = self.loss_func(video_logits, labels_onehot.type(torch.float32))
        self.train_metrics.update(video_pred, labels_onehot)
        self.train_map_head.update(video_pred[:, self.classes_head], labels_onehot[:, self.classes_head])
        self.train_map_middle.update(video_pred[:, self.classes_middle], labels_onehot[:, self.classes_middle])
        self.train_map_tail.update(video_pred[:, self.classes_tail], labels_onehot[:, self.classes_tail])
        self.train_map_class.update(video_pred, labels_onehot)
        on_step = batch_idx
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        _train_metrics = self.train_metrics.compute()
        self.log_dict(_train_metrics)
        self.train_metrics.reset()

        _train_map_head = self.train_map_head.compute()
        self.log("train_map_head", _train_map_head)
        self.train_map_head.reset()

        _train_map_middle = self.train_map_middle.compute()
        self.log("train_map_middle", _train_map_middle)
        self.train_map_middle.reset()

        _train_map_tail = self.train_map_tail.compute()
        self.log("train_map_tail", _train_map_tail)
        self.train_map_tail.reset()

        _train_map_class = self.train_map_class.compute()
        for i in range(self.n_classes):
            self.log('train_map_' + self.class_names[i], _train_map_class[i])
        self.train_map_class.reset()

    def validation_step(self, batch, batch_idx):
        video_logits, labels_onehot = self(batch)
        video_pred = torch.sigmoid(video_logits)
        loss = self.loss_func(video_logits, labels_onehot.type(torch.float32))
        self.valid_metrics.update(video_pred, labels_onehot)
        self.valid_map_head.update(video_pred[:, self.classes_head], labels_onehot[:, self.classes_head])
        self.valid_map_middle.update(video_pred[:, self.classes_middle], labels_onehot[:, self.classes_middle])
        self.valid_map_tail.update(video_pred[:, self.classes_tail], labels_onehot[:, self.classes_tail])
        self.valid_map_class.update(video_pred, labels_onehot)
        self.log("valid_loss", loss)

    def on_validation_epoch_end(self):
        _valid_metrics = self.valid_metrics.compute()
        self.log_dict(_valid_metrics)
        self.valid_metrics.reset()

        _valid_map_head = self.valid_map_head.compute()
        self.log("valid_map_head", _valid_map_head)
        self.valid_map_head.reset()

        _valid_map_middle = self.valid_map_middle.compute()
        self.log("valid_map_middle", _valid_map_middle)
        self.valid_map_middle.reset()

        _valid_map_tail = self.valid_map_tail.compute()
        self.log("valid_map_tail", _valid_map_tail)
        self.valid_map_tail.reset()

        _valid_map_class = self.valid_map_class.compute()
        for i in range(self.n_classes):
            self.log('valid_map_' + self.class_names[i], _valid_map_class[i])
        self.valid_map_class.reset()

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-6, betas=(0.9, 0.98))
        else:
            assert False, f"Unknown optimizer: {optimizer}"

        if self.decay_power == "no_decay":
            return optimizer
        else:
            if self.decay_power == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                )
            elif self.decay_power == "poly":
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=self.end_lr,
                    power=self.poly_decay_power,
                )
            sched = {"scheduler": scheduler, "interval": "step"}

            return ([optimizer], [sched])    
    
# if __name__ == "__main__":    
    # # fast stream
    # from config import config
    # _config = config()
    
    # transformer_fast = TransformerFast(
    #     config['num_frames_fast'], # config['num_frames']
    #     config['transformer_width_fast'],
    #     config['transformer_layers_fast'],
    #     config['transformer_heads_fast']
    # )

    # x = torch.rand(_config['batch_size'], _config['num_frames_fast'], _config['transformer_width_fast'])
    # x = transformer_fast(x)
    # print(x.shape)

    # # Whole Model
    # import numpy as np
    # from torch import utils
    # from config import config
    # from InternVideo import tokenize
    # from Dataset import AnimalKingdomDataset

    # device = 'cpu'
    # _config = config()
    # _config['batch_size'] = 2

    # dataset_train = AnimalKingdomDataset(_config, split="train")
    # dataset_valid = AnimalKingdomDataset(_config, split="val")

    # _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    # model = VideoCLIP(_config)

    # ckpt_fp = os.path.join(os.path.dirname(__file__), "weights", "epoch=2-step=9003.ckpt")
    # if os.path.exists(ckpt_fp):
    #     model.load_ckpt_state_dict(ckpt_fp)

    # dataset_train.produce_prompt_embedding(model.image_clip)
    # dataset_valid.produce_prompt_embedding(model.image_clip)
    # model.set_text_feats(dataset_train.text_features)

    # train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True) # TODO: DEBUG num_workers=4, maybe MACOS bug
    # valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False) # TODO: DEBUG num_workers=4, maybe MACOS bug

    # # # test otptimizer
    # # optimizer = model.configure_optimizers()

    # # # test forward and train
    # # for batch_idx, (video_tensor, labels_onehot) in enumerate(train_loader):
    # #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    # #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    # #     print(loss)
    # #     break

    # # for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
    # #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    # #     model.validation_step((video_tensor, labels_onehot), batch_idx)
    # #     break

    # # test inference
    # for batch_idx, (video_tensor, labels_onehot, index) in enumerate(train_loader):
    #     batch = video_tensor.to(device), labels_onehot.to(device), index
    #     video_logits = model(batch)
    #     video_logits = video_logits.cpu().detach().numpy()
    #     print(video_logits.shape)
    #     # np.save(os.path.join(os.path.dirname(__file__), "temp", "video_logits.npy"), video_logits)
    #     break
    # # video_logits = np.load(os.path.join(os.path.dirname(__file__), "temp", "video_logits.npy"))
    
    # sameple_idx = 1
    # y = np.where(labels_onehot[sameple_idx])[0]
    # y_pred = np.where(video_logits[sameple_idx] > 0.5)[0]

    # df_action = dataset_train.df_action

    # print("Ground Truth:")    
    # for idx, prompt in df_action.loc[y, 'prompt'].items():
    #     print(str(idx).zfill(3) + ":", prompt)
    # print("====================================")

    # print("Prediction:")
    # for idx, prompt in df_action.loc[y_pred, 'prompt'].items():
    #     print(str(idx).zfill(3) + "(%.2f)"%video_logits[sameple_idx][idx] + ":", prompt)
    # print("====================================")

