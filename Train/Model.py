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
from torch.utils.checkpoint import checkpoint
from ModelUtil.clip_param_keys import clip_param_keys
from open_clip import _build_vision_tower, Transformer, LayerNorm
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

class AfricaTransformer(nn.Module):
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

class VideoCLIP(pl.LightningModule):
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

        self.final_fc = torch.nn.Linear(self.n_classes, self.n_classes)
        self.clip, self.clip_preprocess = clip_kc_new.load( # class VideoIntern(nn.Module):
            config["clip"], # /pathto/ViT-L-14.pt
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
        ckpt = torch.load(config["load_path"], map_location="cpu")
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

        if config['train_laryers'] == "vision":
            self.freeze_text()
        if config['train_laryers'] == "vision_proj":
            self.freeze_clip_evl()

        # africa
        self.africa = config['africa']
        if self.africa:
            africa_config_fp = os.path.join(os.path.dirname(__file__), "open_clip/model_configs/ViT-L-14.json")
            with open(africa_config_fp, 'r') as f:
                africa_model_config = json.load(f)
            self.clip_africa = _build_vision_tower(africa_model_config['embed_dim'], africa_model_config['vision_cfg'])
            if config['original_clip_africa']:
                # original_clip
                state_dict_africa = torch.jit.load(config["ckpt_path_africa"], map_location="cpu").visual.state_dict()
            else:
                # pretrained africa clip
                state_dict_africa = torch.load(config["ckpt_path_africa"], map_location="cpu")['state_dict']
                state_dict_africa = {name.replace("image_encoder.", ""): weights for name, weights in state_dict_africa.items() if "image_encoder" in name}
            self.clip_africa.load_state_dict(state_dict_africa)
            self.clip_africa.requires_grad = False
            self.clip_africa.eval()
            self.transformer_africa = AfricaTransformer(
                _config['num_frames_africa'],
                _config['clip_width_africa'],
                _config['clip_layers_africa'],
                _config['clip_heads_africa']
            )
            self.w1_africa = nn.Parameter(torch.randn(_config['clip_width_africa']))
            self.w2_africa = nn.Parameter(torch.randn(_config['clip_width_africa']))
            self.bias = nn.Parameter(torch.randn(_config['clip_width_africa']))

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

    def freeze_clip_evl(self):
        for n, p in self.named_parameters():
            if (
                "clip.visual" in n
                and "clip.visual.ln_post" not in n
                and "clip.visual.proj" not in n
            ):
                p.requires_grad = False
            elif "clip.transformer" in n:
                p.requires_grad = False
            elif "clip.token_embedding" in n:
                p.requires_grad = False
            elif "clip.positional_embedding" in n:
                p.requires_grad = False

    def freeze_clip(self):
        for n, p in self.named_parameters():
            # Unfreeze the projection layer
            if any(x in n for x in ["text_projection", "visual_proj", "visual.proj"]):
                continue
            if any(x in n for x in clip_param_keys):
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "clip.transformer" in n:
                p.requires_grad = False
            elif "clip.token_embedding" in n:
                p.requires_grad = False
            elif "clip.positional_embedding" in n:
                p.requires_grad = False
            elif "clip.ln_final" in n:
                p.requires_grad = False
            elif "clip.text_projection" in n:
                p.requires_grad = False
            elif "clip.eot_token_embedding" in n:
                p.requires_grad = False

    def forward_clip(self, batch):
        video_tensor, video_tensor_africa, labels, index = batch
        video_tensor = video_tensor.contiguous().transpose(1, 2)
        video_feats, video_all_feats = self.clip.encode_video(
            video_tensor, return_all_feats=True, mode='video'
        )
        return video_feats
    
    def forward_clip_africa(self, batch):
        video_tensor, video_tensor_africa, labels, index = batch
        B, F, C, H, W = video_tensor_africa.shape
        video_feats = self.clip_africa(video_tensor_africa.view(B*F, C, H, W))
        video_feats = video_feats.view(B, F, -1)
        return video_feats

    def forward(self, batch):
        video_feats = self.forward_clip(batch)
        if self.africa:
            video_feats_africa = self.forward_clip_africa(batch) # (B, F, 768)
            video_feats_africa = self.transformer_africa(video_feats_africa)
            video_feats = video_feats * self.w1_africa + video_feats_africa * self.w2_africa + self.bias

        video_feats = torch.nn.functional.normalize(video_feats, dim=1) # (n, 768)
        text_feats = torch.nn.functional.normalize(self.text_feats, dim=1) # (140, 768)
        t = self.clip.logit_scale.exp()
        video_logits = ((video_feats @ text_feats.t()) * t)#.softmax(dim=-1) # (n, 140)
        video_logits = self.final_fc(video_logits)
        return video_logits
        
    def training_step(self, batch, batch_idx):
        video_tensor, video_tensor_africa, labels_onehot, index = batch
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
        video_tensor, video_tensor_africa, labels_onehot, index = batch
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
    # AFRICA
    # from config import config
    # _config = config()
    
    # africa_transformer = AfricaTransformer(
    #     _config['num_frames_africa'],
    #     _config['clip_width_africa'],
    #     _config['clip_layers_africa'],
    #     _config['clip_heads_africa']
    #     )
    
    # x = torch.rand(_config['batch_size'], _config['num_frames_africa'], _config['clip_width_africa'])
    # x = africa_transformer(x)
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
    for batch_idx, (video_tensor, video_tensor_africa, labels_onehot, index) in enumerate(train_loader):
        batch = video_tensor.to(device), video_tensor_africa.to(device), labels_onehot.to(device), index
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

