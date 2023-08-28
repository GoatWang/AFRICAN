import os
import re
import cv2
import json
import math
from copy import deepcopy

import torch
import numpy as np
from torch import nn
import torchmetrics
from Loss import get_loss_func
import pytorch_lightning as pl
import InternVideo as clip_kc_new
from typing import Callable, Sequence, Tuple
from open_clip import Transformer, LayerNorm
from torch.utils.checkpoint import checkpoint
from ModelUtil.clip_param_keys import clip_param_keys
from open_clip import CLIPVisionCfg, _build_vision_tower, create_model_and_transforms
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

class TemporalTransformer(nn.Module):
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


class AfricanSlowfast(pl.LightningModule):
    """
    VC: self init logit scale
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

    VC2: use pretrained logit scale
        # w = torch.load("/notebooks/AnimalKingdomCLIP/Train/weights/ViT-L-14.pt")
        # [wei for k, wei in w.named_parameters() if "logit" in k]
        # 4.6052
        # w = torch.load("/notebooks/AnimalKingdomCLIP/Train/weights/InternVideo-MM-L-14.ckpt")
        # [w['state_dict'][k] for k in w['state_dict'].keys() if "logit" in k]
        # 4.6052
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["lr"]
        self.n_classes = config["n_classes"]
        self.optimizer = config["optimizer"]
        self.image_encoder_batch_size = config["image_encoder_batch_size"]
        self.transformer_width_ic = config['transformer_width_ic']
        self.transformer_width_af = config['transformer_width_af']

        self.num_frames = config['num_frames']
        self.decay_power = config['decay_power']
        self.warmup_steps = config['warmup_steps']
        self.max_steps = config['max_steps']
        self.end_lr = config['end_lr']
        self.poly_decay_power = config['poly_decay_power']

        self.final_fc = torch.nn.Linear(self.n_classes, self.n_classes)

        self.enable_video_clip = config['enable_video_clip']
        self.video_clip = self.get_video_clip_model(config) # video encoder (slow stream)
        self.transformer_proj_vc = config['transformer_proj_vc']
        if self.transformer_proj_vc:
            self.transformer_vc = TemporalTransformer(
                self.num_frames,
                config['transformer_width_vc'],
                config['transformer_layers_vc'],
                config['transformer_heads_vc']
            )
            self.w_vc_clstoken = nn.Parameter(torch.randn(config['transformer_width_vc']))
            self.w_ic_features = nn.Parameter(torch.randn(config['transformer_width_vc']))
            self.bias_vc = nn.Parameter(torch.randn(config['transformer_width_vc']))
            
        self.use_text_proj = config['use_text_proj']
        if self.use_text_proj:
            self.text_proj = nn.Linear(config['transformer_width_vc'], config['transformer_width_vc'])

        if config['train_laryers'] == "vision":
            self.freeze_video_clip_text(self.video_clip)
            
        if config['train_laryers'] == "vision_proj":
            self.freeze_video_clip_evl()
            print("freeze_video_clip_evl")

        if config['train_laryers'].startswith("vision_tn"):
            matches = re.findall("vision_tn\d+", config['train_laryers'])
            assert len(matches) > 0, "train_laryers should be one of vision_tn\d+"
            n = int(matches[0].replace("vision_tn", ""))
            self.freeze_video_clip_evl_exclude_tnx(n)
            print(f"freeze last {n} layers of vision transformer")

        if config['train_laryers'] == "vision_dd_proj":
            self.freeze_video_clip_evl_exclude_dd()
            print("freeze_video_clip_evl_exclude_dd")
        # self.print_requires_grad(self.video_clip)
            
        # slowfast: enable fast stream
        self.enable_image_clip = config['enable_image_clip']
        if self.enable_image_clip:
            self.image_encoder_ic = self.get_image_encoder_fast(config, "ic")
            self.freeze_image_encoder_fast_evl(self.image_encoder_ic)
            self.transformer_ic = TemporalTransformer(
                self.num_frames,
                config['transformer_width_ic'],
                config['transformer_layers_ic'],
                config['transformer_heads_ic']
            )
            # to be merged on image encoder featers (transformer_width_ic: 768)
            self.w_vc = nn.Parameter(torch.randn(config['transformer_width_ic']))
            self.w_ic = nn.Parameter(torch.randn(config['transformer_width_ic']))
            self.bias_ic = nn.Parameter(torch.randn(config['transformer_width_ic']))
        
        self.enable_african = config['enable_african']
        if self.enable_african:
            self.image_encoder_af = self.get_image_encoder_fast(config, "af")
            self.freeze_image_encoder_fast_evl(self.image_encoder_af)
            self.transformer_af = TemporalTransformer(
                self.num_frames,
                config['transformer_width_af'],
                config['transformer_layers_af'],
                config['transformer_heads_af']
            )

            output_width = int(config['transformer_width_af'])
            self.mlp_af = nn.Sequential(
                nn.Linear(output_width, output_width//2),
                nn.Linear(output_width//2, output_width//4),
                nn.Linear(output_width//4, self.n_classes)
            )

            # to be merged on final layers (self.n_classes)
            self.w_vcic = nn.Parameter(torch.ones(self.n_classes) * 0.9)
            self.w_af = nn.Parameter(torch.ones(self.n_classes) * 0.1)
            self.bias_af = nn.Parameter(torch.randn(self.n_classes))

    def print_requires_grad(self, model):
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

    def get_video_clip_model(self, config):
        clip, clip_preprocess = clip_kc_new.load( # class VideoIntern(nn.Module):
            config["ckpt_path_imageclip_vc"], # /pathto/ViT-L-14.pt
            t_size=config["num_frames"], # num_frames=8
            n_layers=4,
            mlp_dropout=[config["clip_evl_dropout"]] * 4, # 0.0
            cls_dropout=config["clip_evl_dropout"], # 0.0
            no_pretrain=config["clip_no_pretrain"], # False
            init_zero=config["clip_init_zero"], # True
            drop_path_rate=config["clip_dpr"], # 0.0
            device=self.device,
            use_checkpoint=config["clip_use_checkpoint"],
            checkpoint_num=config["clip_checkpoint_num"],
        )
        ckpt = torch.load(config["ckpt_path_videoclip_vc"], map_location="cpu")
        state_dict = ckpt["state_dict"]

        sd = {k: v.cpu() for k, v in self.state_dict().items()}
        for k in list(state_dict.keys()):
            if k not in sd:
                continue
            if state_dict[k].shape != sd[k].shape:
                print(
                    "!!!!!!!!!!!Size mismatch {} {} {}".format(
                        k, state_dict[k].shape, sd[k].shape
                    )
                )
                del state_dict[k]

        self.load_state_dict(state_dict, strict=False)
        return clip
    
    def get_image_encoder_fast(self, config, pretrained_type):
        """
        pretrained_type: {"ic", "af"}
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
        if pretrained_type == "ic":
            # original_clip
            state_dict_ic = torch.jit.load(config['ckpt_path_ic'], map_location="cpu").visual.state_dict()
            image_encoder.load_state_dict(state_dict_ic)
        elif pretrained_type == "af":
            # pretrained african
            state_dict_af = torch.load(config['ckpt_path_af'], map_location="cpu")['state_dict']
            state_dict_af = {name.replace("image_encoder.", ""): weights for name, weights in state_dict_af.items() if "image_encoder" in name}
            image_encoder.load_state_dict(state_dict_af)
        else:
            raise ValueError("pretrained_type should be one of {ic, af}")
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

    def freeze_video_clip_evl(self):
        for n, p in self.video_clip.named_parameters():
            if (
                "visual" in n
                and "visual_ln_post" not in n # and "visual_ln_post" not in n "visual.ln_post"
                and "visual_proj" not in n # and "visual_proj" not in n "visual.proj"
            ):
                p.requires_grad = False
            elif "transformer" in n:
                p.requires_grad = False
            elif "token_embedding" in n:
                p.requires_grad = False
            elif "positional_embedding" in n:
                p.requires_grad = False

    def freeze_video_clip_evl_exclude_tnx(self, n=4):
        '''
        exclude last 4 layers in vision transformer
        '''
        assert n <= 24, "n should be less than 24"
        self.freeze_video_clip_evl()
        for resblock in self.video_clip.visual.transformer.resblocks[-n:]:
            resblock.requires_grad_(True)
        self.video_clip.visual.transformer.dpe.requires_grad_(True)
        self.video_clip.visual.transformer.dec.requires_grad_(True)

    def freeze_video_clip_evl_exclude_dd(self):
        '''
        exclude dpe and dec layers in vision transformer
        '''
        self.freeze_video_clip_evl()
        self.video_clip.visual.transformer.dpe.requires_grad_(True)
        self.video_clip.visual.transformer.dec.requires_grad_(True)

    def freeze_video_clip_text(self, model):
        for n, p in model.named_parameters():
            if "transformer" in n:
                p.requires_grad = False
            elif "token_embedding" in n:
                p.requires_grad = False
            elif "positional_embedding" in n:
                p.requires_grad = False
            elif "ln_final" in n:
                p.requires_grad = False
            elif "text_projection" in n:
                p.requires_grad = False
            elif "eot_token_embedding" in n:
                p.requires_grad = False

    def freeze_image_encoder_fast_evl(self, model):
        """
        these layers upadted: class_embedding, proj, ln_post.weight, ln_post.bias
        """
        for n, p in model.named_parameters():
            if ( # no layers coms in
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

    def forward_frames_vc(self, frames_tensor):
        frames_tensor = frames_tensor.contiguous().transpose(1, 2)

        # cls token (Intern Video Setting)
        frames_feats, video_all_feats = self.video_clip.encode_video(
            frames_tensor, return_all_feats=True, mode='video'
        )

        if self.transformer_proj_vc:
            # all feature for running temporal transformer
            x_feats = video_all_feats[1:]
            x_feats = x_feats.permute(1, 2, 0, 3)
            B, F, L, W = x_feats.shape
            x_feats = x_feats.reshape(-1, W) @ self.video_clip.visual_proj
            x_feats = x_feats.reshape(B, F, L, x_feats.shape[-1])
            x_feats = torch.sum(x_feats, dim=2)
            x_feats = self.transformer_vc(x_feats)

            # weighted sum
            frames_feats = frames_feats * self.w_vc_clstoken + x_feats * self.w_ic_features + self.bias_vc
        return frames_feats

    # # memory efficiency: not in use
    # def forward_frames_vc_memef(self, frames_tensor):
    #     frames_tensor = frames_tensor.contiguous().transpose(1, 2)
    #     with torch.no_grad():
    #         feats_tensor = torch.zeros(frames_tensor.shape[0], self.transformer_width_vc, device=self.device)
    #         n_iters = int(np.ceil(frames_tensor.shape[0] / self.preprocess_batch_size_vc))
    #         for idx in range(n_iters):
    #             st, end = idx*self.preprocess_batch_size_vc, (idx+1)*self.preprocess_batch_size_vc
    #             feats_tensor[st:end] = self.video_clip.visual(frames_tensor[st:end], return_all_feats=False, mode='video')

    #     feats_tensor = self.video_clip.visual_ln_post(feats_tensor)
    #     feats_tensor = feats_tensor @ self.video_clip.visual_proj
    #     return feats_tensor    
    
    # def forward_frames_ic(self, frames_tensor):
    #     """encode image into embedding"""
    #     B, F, C, H, W = frames_tensor.shape
    #     frames_tensor = frames_tensor.view(B*F, C, H, W)
    #     frames_feats = self.image_encoder_ic(frames_tensor)
    #     frames_feats = frames_feats.view(B, F, -1)
    #     frames_feats = self.forward_frames_feats_ic(frames_feats)
    #     return frames_feats
    
    # memory efficiency
    def forward_frames_memef_ic(self, frames_tensor):
        with torch.no_grad():
            B, F, C, H, W = frames_tensor.shape
            frames_tensor = frames_tensor.reshape(B*F, C, H, W)
            frames_feats = torch.zeros(B*F, self.transformer_width_ic, device=self.device)
            n_iters = int(np.ceil(frames_feats.shape[0] / self.image_encoder_batch_size))
            for idx in range(n_iters):
                st, end = idx*self.image_encoder_batch_size, (idx+1)*self.image_encoder_batch_size
                frames_feats[st:end] = self.image_encoder_ic(frames_tensor[st:end])
            frames_feats = frames_feats.reshape(B, F, self.transformer_width_ic)
        frames_feats = self.forward_frames_feats_ic(frames_feats)
        return frames_feats
    
    def forward_frames_feats_ic(self, frames_feats):
        """apply transformer on image embedding of each frames"""
        frames_feats = self.transformer_ic(frames_feats)
        return frames_feats

    # def forward_frames_af(self, frames_tensor):
    #     """encode image into embedding"""
    #     B, F, C, H, W = frames_tensor.shape
    #     frames_tensor = frames_tensor.view(B*F, C, H, W)
    #     frames_feats = self.image_encoder_af(frames_tensor)
    #     frames_feats = frames_feats.view(B, F, -1)
    #     frames_feats = self.forward_frames_feats_af(frames_feats)
    #     return frames_feats
    
    # memory efficiency
    def forward_frames_memef_af(self, frames_tensor):
        with torch.no_grad():
            B, F, C, H, W = frames_tensor.shape
            frames_tensor = frames_tensor.reshape(B*F, C, H, W)
            frames_feats = torch.zeros(B*F, self.transformer_width_af, device=self.device)
            n_iters = int(np.ceil(frames_feats.shape[0] / self.image_encoder_batch_size))
            for idx in range(n_iters):
                st, end = idx*self.image_encoder_batch_size, (idx+1)*self.image_encoder_batch_size
                frames_feats[st:end] = self.image_encoder_af(frames_tensor[st:end])
            frames_feats = frames_feats.reshape(B, F, self.transformer_width_af)
        frames_feats = self.forward_frames_feats_af(frames_feats)
        return frames_feats

    def forward_frames_feats_af(self, frames_feats):
        """apply transformer on image embedding of each frames"""
        frames_feats = self.transformer_af(frames_feats)
        return frames_feats

    def forward(self, batch):
        frames_tensor, labels_onehot, index = batch

        # enable_video_clip
        frames_feats = None
        if self.enable_video_clip:
            frames_feats = self.forward_frames_vc(frames_tensor)

        # enable_image_clip        
        if self.enable_image_clip:
            frames_feats_ic = self.forward_frames_memef_ic(frames_tensor)
            if frames_feats is None:
                frames_feats = frames_feats_ic
            else:
                frames_feats = frames_feats * self.w_vc + frames_feats_ic * self.w_ic + self.bias_ic

        video_logits_vcic = None
        if frames_feats is not None:
            text_feats = self.text_feats
            if self.use_text_proj:
                text_feats = self.text_proj(self.text_feats)

            video_feats = torch.nn.functional.normalize(frames_feats, dim=1) # (n, 768)
            text_feats = torch.nn.functional.normalize(text_feats, dim=1) # (140, 768)
            t = self.video_clip.logit_scale.exp()
            video_logits_vcic = ((video_feats @ text_feats.t()) * t)#.softmax(dim=-1) # (n, 140)
            video_logits_vcic = self.final_fc(video_logits_vcic)

        # enable_african
        video_logits_af = None
        if self.enable_african:
            frames_feats_af = self.forward_frames_memef_af(frames_tensor)
            video_logits_af = self.mlp_af(frames_feats_af)

        if video_logits_vcic is None:
            video_logits = video_logits_af
        elif video_logits_af is None:
            video_logits = video_logits_vcic
        else:
            video_logits = video_logits_vcic * self.w_vcic + video_logits_af * self.w_af + self.bias_af

        return video_logits_vcic, video_logits_af, video_logits, labels_onehot

    # def infer(self, frames_tensor, rames_tensor_fast):
    #     frames_tensor, frames_tensor_fast, labels_onehot, index = batch
    #     frames_feats_slow = self.forward_frames_slow(frames_tensor)
    #     frames_feats_fast = self.forward_frames_fast(frames_tensor_fast)
    #     frames_feats = frames_feats_slow * self.w_slow + frames_feats_fast * self.w_fast + self.bias
    #     video_feats = torch.nn.functional.normalize(frames_feats, dim=1) # (n, 768)
    #     text_feats = torch.nn.functional.normalize(self.text_feats, dim=1) # (140, 768)
    #     t = self.video_clip.logit_scale.exp()
    #     video_logits = ((video_feats @ text_feats.t()) * t)#.softmax(dim=-1) # (n, 140)
    #     video_logits = self.final_fc(video_logits)
    #     return video_logits
    
    def training_step(self, batch, batch_idx):
        video_logits_vcic, video_logits_af, video_logits, labels_onehot = self(batch)

        if (video_logits_vcic is None) and (video_logits_af is None):
            assert False, "video_logits_vcic and video_logits_af are both None"
        elif (video_logits_vcic is None) or (video_logits_af is None): # one of them is None
            loss = self.loss_func(video_logits, labels_onehot.type(torch.float32))
            self.log("train_loss", loss)
        else: # both of them are not None
            loss_vcic = self.loss_func(video_logits_vcic, labels_onehot.type(torch.float32))
            loss_af = self.loss_func(video_logits_af, labels_onehot.type(torch.float32))
            loss_all = self.loss_func(video_logits, labels_onehot.type(torch.float32))
            loss = (loss_vcic + loss_af + loss_all) / 3
            self.log("train_loss_vcic", loss_vcic)
            self.log("train_loss_af", loss_af)
            self.log("train_loss", loss_all)

        video_pred = torch.sigmoid(video_logits)
        self.train_metrics.update(video_pred, labels_onehot)
        self.train_map_head.update(video_pred[:, self.classes_head], labels_onehot[:, self.classes_head])
        self.train_map_middle.update(video_pred[:, self.classes_middle], labels_onehot[:, self.classes_middle])
        self.train_map_tail.update(video_pred[:, self.classes_tail], labels_onehot[:, self.classes_tail])
        self.train_map_class.update(video_pred, labels_onehot)
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
        video_logits_vcic, video_logits_af, video_logits, labels_onehot = self(batch)

        if (video_logits_vcic is None) and (video_logits_af is None):
            assert False, "video_logits_vcic and video_logits_af are both None"
        elif (video_logits_vcic is None) or (video_logits_af is None): # one of them is None
            loss = self.loss_func(video_logits, labels_onehot.type(torch.float32))
            self.log("valid_loss", loss)
        else: # both of them are not None
            loss_vcic = self.loss_func(video_logits_vcic, labels_onehot.type(torch.float32))
            loss_af = self.loss_func(video_logits_af, labels_onehot.type(torch.float32))
            loss_all = self.loss_func(video_logits, labels_onehot.type(torch.float32))
            loss = (loss_vcic + loss_af + loss_all) / 3
            self.log("valid_loss_vcic", loss_vcic)
            self.log("valid_loss_af", loss_af)
            self.log("valid_loss", loss_all)

        video_pred = torch.sigmoid(video_logits)
        self.valid_metrics.update(video_pred, labels_onehot)
        self.valid_map_head.update(video_pred[:, self.classes_head], labels_onehot[:, self.classes_head])
        self.valid_map_middle.update(video_pred[:, self.classes_middle], labels_onehot[:, self.classes_middle])
        self.valid_map_tail.update(video_pred[:, self.classes_tail], labels_onehot[:, self.classes_tail])
        self.valid_map_class.update(video_pred, labels_onehot)

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
    

    def forward_video_encoder_att_map(self, video):
        visual = self.video_clip.visual

        x = video
        x = x.contiguous().transpose(1, 2)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]

        # prepare patches, cls embedding and positional embedding
        N, C, T, H, W = x.shape # bs, width, num_frames, grid_rows, grid_cols
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C) # [1*8, 16*16, 1024], 8 frames/video, 16*16 patches/frame, 1024 dim/patch
        #              1024                               for boradcast 8            1024                                         (8, 256, 1024) 
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] # [1*8, 257, 1024]
        x = x + visual.positional_embedding.to(x.dtype) # [1*8, 257, 1024]
        x = visual.ln_pre(x) # [1*8, 257, 1024]
        x = x.permute(1, 0, 2)  # NLD -> LND  # [257, 8, 1024]

        # transformer
        transformer = visual.transformer
        attn_output_weights_layers = []
        for i, resblock in enumerate(transformer.resblocks):
            x = resblock(x, self.num_frames) # 257, 8, 1024

            x_in = x
            x = resblock.ln_1(x)
            q, k, x, attn_output_weights = resblock.attn(x, x, x, need_weights=True, return_qk=True)
            x = x_in + resblock.drop_path(x)

            x_in = x
            x = resblock.ln_2(x)
            x = resblock.mlp(x)
            x = x_in + resblock.drop_path(x)

            attn_output_weights_layers.append(attn_output_weights)
            
        attn_output_weights_layers = torch.stack(attn_output_weights_layers)
        return attn_output_weights_layers


    def forward_image_encoder_att_map(self, image, image_encoder, return_attn_only=True):
        """
        # video_tensors, labels_onehot = next(iter(valid_loader))
        # video_tensor1, video_tensor2, labels_onehot = video_tensors[0].to(_config['device']), video_tensors[1].to(_config['device']), labels_onehot.to(_config['device'])

        # x1 = video_tensor1[:, 0] # 3, 224, 224
        # x2 = video_tensor1[:, 1] # 3, 224, 224
        # print(x1.shape)
        # print(x2.shape)

        # pooled, tokens, attn_output_weights_layers = forward_for_visual(image_encoder, x1)
        # print("len(attn_output_weights_layers)", len(attn_output_weights_layers))
        # print("attn_output_weights_layers[0].shape", attn_output_weights_layers[0].shape)

        return attn_output_weights_layers.shape = torch.Size([24, 8, 16, 257, 257])
        """
        x = image
        x = image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + image_encoder.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = image_encoder.patch_dropout(x)
        x = image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND = [8, 257, 1024] -> [257, 8, 1024]
        
        model_transformer = image_encoder.transformer
        attn_output_weights_layers = []
        for resblock in model_transformer.resblocks:
            # resblock.attention(q_x=resblock.ln_1(x), k_x=None, v_x=None, attn_mask=None)
            # self.attn(x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
            q_x = resblock.ln_1(x)
            x_attn, attn_output_weights = resblock.attn(q_x, q_x, q_x, need_weights=True, average_attn_weights=False, attn_mask=None)
            x = q_x + resblock.ls_1(x_attn)
            x = x + resblock.ls_2(resblock.mlp(resblock.ln_2(x)))
            attn_output_weights_layers.append(attn_output_weights)
        attn_output_weights_layers = torch.stack(attn_output_weights_layers)
        attn_output_weights_layers = torch.mean(attn_output_weights_layers, dim=2)
            
        if return_attn_only:
            return attn_output_weights_layers # torch.Size([24, 8, 16, 257, 257])
        else:
            x = x.permute(1, 0, 2)  # LND -> NLD
            pooled, tokens = image_encoder._global_pool(x)
            pooled = image_encoder.ln_post(pooled)
            pooled = pooled @ image_encoder.proj
            return pooled, tokens, attn_output_weights_layers

    def draw_att_map(self, video_raw, video_aug, encoder_type="ic"):
        """use AnimalKingdomDatasetVisualize Dataset to get attention map"""
        B, F, C, H, W = video_aug.shape
        _, _, _, H_src, W_src = video_raw.shape
        images_raw = video_raw.detach().cpu().numpy().reshape(B*F, C, H_src, W_src).transpose(0, 2, 3, 1)
        images_raw = np.stack([cv2.resize(image_raw, (H, W)) for image_raw in images_raw])
        
        with torch.no_grad():
            if encoder_type == "vc":
                video_aug = video_aug.to(self.device)
                att_mat = self.forward_video_encoder_att_map(video_aug)
            elif encoder_type == "ic":
                images_aug = video_aug.reshape(B*F, C, H, W).to(self.device)
                att_mat = self.forward_image_encoder_att_map(images_aug, self.image_encoder_ic)
            elif encoder_type == "af":
                images_aug = video_aug.reshape(B*F, C, H, W).to(self.device)
                att_mat = self.forward_image_encoder_att_map(images_aug, self.image_encoder_af)
            else:
                raise NotImplementedError

        # att_mat.shape = Layers, B*F, H, W = torch.Size([24, 8, 257, 257])
        # Average the attention weights across all heads.
        att_mat = att_mat.permute(1, 0, 2, 3) # 8, 24 ,257, 257

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(2)).to(self.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).to(self.device)
        joint_attentions[:, 0] = aug_att_mat[:, 0]
        for n in range(1, aug_att_mat.size(1)):
            joint_attentions[:, n] = torch.bmm(aug_att_mat[:, n], joint_attentions[:, n-1])
        
        # generate attn map
        bs = aug_att_mat.shape[0]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        attn_maps = joint_attentions[:, -1, 0, 1:]
        attn_maps = attn_maps / torch.max(attn_maps, dim=1)[0].unsqueeze(-1)
        attn_maps = attn_maps.reshape(bs, grid_size, grid_size).detach().cpu().numpy()

        # generate heatmap
        heatmaps = np.zeros_like(images_raw)
        for i in range(images_raw.shape[0]):
            image_raw = images_raw[i]
            attn_map = cv2.resize(attn_maps[i], image_raw.shape[:2])
            heatmap = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(image_raw, 0.3, heatmap, 0.7, 0.0)
            heatmaps[i] = heatmap

        # batch_size first
        images_raw = images_raw.reshape(B, F, *images_raw.shape[-3:])
        attn_maps = attn_maps.reshape(B, F, *attn_maps.shape[-2:])
        heatmaps = heatmaps.reshape(B, F, *heatmaps.shape[-3:])

        return images_raw, attn_maps, heatmaps

    # def get_attn_map_vc(self, frames_feats):
    # def get_attn_map_ic(self, frames_feats):
    # def get_attn_map_af(self, frames_feats):

if __name__ == "__main__":    
    # # fast stream
    # from config import config
    # _config = config()
    
    # transformer_fast = TemporalTransformer(
    #     config['num_frames_fast'], # config['num_frames']
    #     config['transformer_width_fast'],
    #     config['transformer_layers_fast'],
    #     config['transformer_heads_fast']
    # )

    # x = torch.rand(_config['batch_size'], _config['num_frames_fast'], _config['transformer_width_fast'])
    # x = transformer_fast(x)
    # print(x.shape)

    # Whole Model
    import numpy as np
    from torch import utils
    from config import config
    from InternVideo import tokenize
    from Dataset import AnimalKingdomDataset

    device = 'cpu'
    _config = config()
    _config['batch_size'] = 2

    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = VideoCLIP(_config)

    ckpt_fp = os.path.join(os.path.dirname(__file__), "weights", "epoch=2-step=9003.ckpt")
    if os.path.exists(ckpt_fp):
        model.load_ckpt_state_dict(ckpt_fp)

    dataset_train.produce_prompt_embedding(model.video_clip)
    dataset_valid.produce_prompt_embedding(model.video_clip)
    model.set_text_feats(dataset_train.text_features)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True)
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False)

    # # test otptimizer
    # optimizer = model.configure_optimizers()

    # # test forward and train
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(train_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    #     print(loss)
    #     break

    # for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     model.validation_step((video_tensor, labels_onehot), batch_idx)
    #     break

    # test inference
    for batch_idx, (video_tensor, labels_onehot, index) in enumerate(train_loader):
        batch = video_tensor.to(device), labels_onehot.to(device), index
        video_logits = model(batch)
        video_logits = video_logits.cpu().detach().numpy()
        print(video_logits.shape)
        # np.save(os.path.join(os.path.dirname(__file__), "temp", "video_logits.npy"), video_logits)
        break
    # video_logits = np.load(os.path.join(os.path.dirname(__file__), "temp", "video_logits.npy"))
    
    sameple_idx = 1
    y = np.where(labels_onehot[sameple_idx])[0]
    y_pred = np.where(video_logits[sameple_idx] > 0.5)[0]

    df_action = dataset_train.df_action

    print("Ground Truth:")    
    for idx, prompt in df_action.loc[y, 'prompt'].items():
        print(str(idx).zfill(3) + ":", prompt)
    print("====================================")

    print("Prediction:")
    for idx, prompt in df_action.loc[y_pred, 'prompt'].items():
        print(str(idx).zfill(3) + "(%.2f)"%video_logits[sameple_idx][idx] + ":", prompt)
    print("====================================")

