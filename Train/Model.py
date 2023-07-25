import os
import json
import math
from copy import deepcopy

import torch
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

class TransformerFast(nn.Module):
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
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["lr"]
        self.n_classes = config["n_classes"]
        self.optimizer = config["optimizer"]

        self.decay_power = config['decay_power']
        self.warmup_steps = config['warmup_steps']
        self.max_steps = config['max_steps']
        self.end_lr = config['end_lr']
        self.poly_decay_power = config['poly_decay_power']

        self.slowfast = config['slowfast']
        self.diff_sampling_fast = config['diff_sampling_fast']
        self.enable_preprocess_fast = config['enable_preprocess_fast']

        self.final_fc = torch.nn.Linear(self.n_classes, self.n_classes)
        self.video_clip = self.get_video_clip_model(config) # video encoder (slow stream)
        if config['train_laryers'] == "vision":
            self.freeze_video_clip_text(self.video_clip)
        if config['train_laryers'] == "vision_proj":
            self.freeze_video_clip_evl(self.video_clip)
            # print("freeze_video_clip_evl")
            # self.print_requires_grad(self.video_clip)
            
        # slowfast: enable fast stream
        if self.slowfast:
            if not self.enable_preprocess_fast:
                self.image_encoder_fast = self.get_image_encoder_fast(config) # image encoder (fast stream)
                self.freeze_image_encoder_fast_evl(self.image_encoder_fast)
                # print("freeze_image_encoder_fast_evl")
                # self.print_requires_grad(self.image_encoder_fast)

            num_frames_fast = config['num_frames_fast'] if not self.diff_sampling_fast else config['num_frames']
            self.transformer_fast = TransformerFast(
                num_frames_fast,
                config['transformer_width_fast'],
                config['transformer_layers_fast'],
                config['transformer_heads_fast']
            )
            self.w_slow = nn.Parameter(torch.randn(config['transformer_width_fast']))
            self.w_fast = nn.Parameter(torch.randn(config['transformer_width_fast']))
            self.bias = nn.Parameter(torch.randn(config['transformer_width_fast']))

    def print_requires_grad(self, model):
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

    def get_video_clip_model(self, config):
        clip, clip_preprocess = clip_kc_new.load( # class VideoIntern(nn.Module):
            config["ckpt_path_imageclip"], # /pathto/ViT-L-14.pt
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
        ckpt = torch.load(config["ckpt_path_videoclip"], map_location="cpu")
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
    
    def get_image_encoder_fast(self, config):
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
        if config['ckpt_path_fast']:
            # original_clip
            state_dict_clip = torch.jit.load(config['ckpt_path_fast'], map_location="cpu").visual.state_dict()
            image_encoder.load_state_dict(state_dict_clip)
        else:
            # pretrained african
            state_dict_african = torch.load(config['ckpt_path_fast'], map_location="cpu")['state_dict']
            state_dict_african = {name.replace("image_encoder.", ""): weights for name, weights in state_dict_african.items() if "image_encoder" in name}
            image_encoder.load_state_dict(state_dict_african)
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

    def freeze_video_clip_evl(self, model):
        for n, p in model.named_parameters():
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

    def freeze_clip(self):
        for n, p in self.named_parameters():
            # Unfreeze the projection layer
            if any(x in n for x in ["text_projection", "visual_proj", "visual.proj"]):
                continue
            if any(x in n for x in clip_param_keys):
                p.requires_grad = False

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
                
    def forward_frames_slow(self, frames_tensor):
        frames_tensor = frames_tensor.contiguous().transpose(1, 2)
        frames_feats, video_all_feats = self.clip.encode_video(
            frames_tensor, return_all_feats=True, mode='video'
        )
        return frames_feats
    
    def forward_frames_fast(self, frames_tensor):
        """encode image into embedding"""
        B, F, C, H, W = frames_tensor.shape
        frames_tensor = frames_tensor.view(B*F, C, H, W)
        frames_feats = self.image_encoder_fast(frames_tensor)
        frames_feats = frames_feats.view(B, F, -1)
        frames_feats = self.forward_frames_feats_fast(self, frames_feats)
        return frames_feats

    def forward_frames_feats_fast(self, frames_feats):
        """apply transformer on image embedding of each frames"""
        frames_feats = self.transformer_fast(frames_feats)
        return frames_feats

    def forward(self, batch):
        if not self.slowfast:
            frames_tensor, labels, index = batch
            frames_feats = self.forward_frames_slow(frames_tensor)
        else:
            if not self.diff_sampling_fast:
                frames_tensor, labels, index = batch
                frames_feats_slow = self.forward_frames_slow(frames_tensor)
                frames_feats_fast = self.forward_frames_fast(frames_tensor)
            else:
                if not self.enable_preprocess_fast:
                    frames_tensor, frames_tensor_fast, labels_onehot, index = batch
                    frames_feats_slow = self.forward_frames_slow(frames_tensor)
                    frames_feats_fast = self.forward_frames_fast(frames_tensor_fast)
                else:
                    frames_tensor, frames_feats_tensor_fast, labels_onehot, index = batch
                    frames_feats_slow = self.forward_frames_slow(frames_tensor)
                    frames_feats_fast = self.forward_frames_feats_fast(frames_feats_tensor_fast)
            frames_feats = frames_feats_slow * self.w_slow + frames_feats_fast * self.w_fast + self.bias

        video_feats = torch.nn.functional.normalize(frames_feats, dim=1) # (n, 768)
        text_feats = torch.nn.functional.normalize(self.text_feats, dim=1) # (140, 768)
        t = self.clip.logit_scale.exp()
        video_logits = ((video_feats @ text_feats.t()) * t)#.softmax(dim=-1) # (n, 140)
        video_logits = self.final_fc(video_logits)
        return video_logits
        
    def training_step(self, batch, batch_idx):
        video_tensor, labels_onehot, index = batch
        video_logits = self(batch)
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
        video_tensor, labels_onehot, index = batch
        video_logits = self(batch)
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
    
if __name__ == "__main__":    
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

    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)
    model.set_text_feats(dataset_train.text_features)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True) # TODO: DEBUG num_workers=4, maybe MACOS bug
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False) # TODO: DEBUG num_workers=4, maybe MACOS bug

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

