import os
import math
from copy import deepcopy

import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
import InternVideo as clip_kc_new
from ModelUtil.clip_param_keys import clip_param_keys
# from CoTrain.modules import heads, cotrain_utils
# from CoTrain.modules import objectives as objectives
# from CoTrain.modules import base_vision_transformer as vit
# from CoTrain.modules.text_prompt import text_prompt

# def vis_save(imgs, texts):
#     # img: [B, T, C, H, W]
#     # texts: [str]
#     os.makedirs("vis_test", exist_ok=True)
#     imgs = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()
#     imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
#     for img, text in zip(imgs, texts):
#         caption = "_".join(text.split())
#         os.makedirs(os.path.join("vis_test", caption), exist_ok=True)
#         for i, im in enumerate(img):
#             img_path = os.path.join("vis_test", caption, f"{i}.png")
#             plt.imsave(img_path, im)


# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [
#         torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


# @torch.no_grad()
# def batch_shuffle_ddp(x):
#     """
#     Batch shuffle, for making use of BatchNorm.
#     *** Only support DistributedDataParallel (DDP) model. ***
#     """
#     # gather from all gpus
#     batch_size_this = x.shape[0]
#     x_gather = concat_all_gather(x)
#     batch_size_all = x_gather.shape[0]

#     num_gpus = batch_size_all // batch_size_this

#     # random shuffle index
#     idx_shuffle = torch.randperm(batch_size_all).cuda()

#     # broadcast to all gpus
#     torch.distributed.broadcast(idx_shuffle, src=0)

#     # index for restoring
#     idx_unshuffle = torch.argsort(idx_shuffle)

#     # shuffled index for this gpu
#     gpu_idx = torch.distributed.get_rank()
#     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

#     return x_gather[idx_this], idx_unshuffle


# @torch.no_grad()
# def batch_unshuffle_ddp(x, idx_unshuffle):
#     """
#     Undo batch shuffle.
#     *** Only support DistributedDataParallel (DDP) model. ***
#     """
#     # gather from all gpus
#     batch_size_this = x.shape[0]
#     x_gather = concat_all_gather(x)
#     batch_size_all = x_gather.shape[0]

#     num_gpus = batch_size_all // batch_size_this

#     # restored index for this gpu
#     gpu_idx = torch.distributed.get_rank()
#     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

#     return x_gather[idx_this]

class VideoCLIP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config["lr"]
        self.optimizer = config["optimizer"]
        self.train_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=config['n_classes'])
        self.valid_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=config['n_classes'])

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

        # self.freeze_clip_evl()
        # self.freeze_clip()
        self.freeze_text()

    def set_text_feats(self, text_feats):
        self.text_feats = torch.tensor(text_feats)

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

    def forward(self, batch, mode="video"):
        video_tensor, labels = batch
        video_tensor = video_tensor.contiguous().transpose(1, 2)
        video_feats, video_all_feats = self.clip.encode_video(
            video_tensor, return_all_feats=True, mode=mode
        )
        video_feats = torch.nn.functional.normalize(video_feats, dim=1)
        text_feats = torch.nn.functional.normalize(self.text_feats, dim=1)
        t = self.clip.logit_scale.exp()
        video_logits = ((video_feats @ text_feats.t()) * t).softmax(dim=-1)
        return video_logits
        
    def training_step(self, batch, batch_idx):
        video_tensor, labels_onehot = batch
        video_logits = self(batch)
        loss = F.cross_entropy(video_logits, labels_onehot)
        self.train_acc(video_logits, labels_onehot)
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video_tensor, labels_onehot = batch
        video_logits = self(batch)
        loss = F.cross_entropy(video_logits, labels_onehot)
        self.valid_acc(video_logits, labels_onehot)
        self.log("valid_loss", loss)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # TODO: add parameter groups
        # TODO: add lr scheduler
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-6, betas=(0.9, 0.98))
        else:
            assert False, f"Unknown optimizer: {optimizer}"
        return optimizer
    
    


    # def infer(
    #     self,
    #     batch,
    #     mask_text=False,
    #     mask_video=False,
    #     input_video_only=False,
    #     input_text_only=False,
    #     caption=False,
    #     mode="video",
    # ):

    #     # Check configs
    #     assert not input_video_only
    #     assert not input_text_only
    #     if mask_text:
    #         assert self.clip_type in ["ori", "evl", "kc", "kc_new"]
    #     if mask_video:
    #         assert self.clip_type in ["ori", "kc", "kc_new"]

    #     text_ids, special_tokens_mask = batch["clip_text_ids"], batch["clip_special_tokens_mask"]
    #     text_feats, text_all_feats = self.clip.encode_text(
    #         text_ids, return_all_feats=True
    #     )

    #     video = batch["video"] # [0]
    #     video = video.contiguous().transpose(1, 2)
    #     video_feats, video_all_feats = self.clip.encode_video(
    #         video, return_all_feats=True, mode=mode
    #     )

    #     ret = {
    #         "video": video,  # N, C, T, H, W
    #         "text_feats": text_feats,  # N, C
    #         "video_feats": video_feats,  # N, C
    #         "text_ids": text_ids,  # N, L
    #         "special_tokens_mask": special_tokens_mask,  # N, L
    #     }

    #     return ret

    # def forward(self, batch, batch_idx=None, mode="video"):
    #     # self.sanity_check()
    #     with torch.no_grad():
    #         self.clip.logit_scale.clamp_(0, math.log(100))

    #     ret = dict()
    #     ret.update(self.infer(batch, mode=mode))

    #     # if "contrastive" in self.current_tasks:
    #     #     ret.update(objectives.compute_contrastive(self, batch, mode=mode))

    #     # if "multiple_choice" in self.current_tasks:
    #     #     ret.update(objectives.compute_multiple_choice(self, batch))

    #     # if "zs_classify" in self.current_tasks:
    #     #     if self.text_ret is None:
    #     #         # print(f"Generate text features for in batch-{batch_idx}")
    #     #         self.text_ret = self.forward_text()
    #     #     ret.update(objectives.compute_zs_classify(self, batch, self.text_ret))

    #     return ret

    # def forward_text(self):
    #     classes, num_text_aug, _ = text_prompt(prompt_type=self.prompt_type)
    #     text_inputs = classes.to(self.device)
    #     text_feats = self.clip.encode_text(text_inputs)
    #     # text_feats /= text_feats.norm(dim=-1, keepdim=True)

    #     ret = {
    #         "text_feats": text_feats,  # num_text_aug * num_classes, C
    #         "num_text_aug": num_text_aug,
    #     }
    #     return ret

    # def forward_video(self, batch):
    #     img = batch["video"][0]
    #     if self.clip_type in ["ori", "evl", "kc", "kc_new"]:
    #         # [B, T, C, H, W] -> [B, C, T, H, W]
    #         img = img.contiguous().transpose(1, 2)
    #     video_feats = self.clip.encode_video(img)

    #     ret = {
    #         "video_feats": video_feats,  # N, C
    #     }
    #     return ret

    # def training_step(self, batch, batch_idx):
    #     # gradually_freeze_by_layer(self, self.global_step, self.grad_unfreeze_int)
    #     cotrain_utils.set_task(self)
    #     # self.momentum_checkpoint()
    #     # co-training
    #     if "v" in batch and "i" in batch:
    #         video_output, image_output = {}, {}
    #         if not self.alt_data or batch_idx % 2 == 0:
    #             video_output = self(batch["v"], mode="video")
    #         if not self.alt_data or batch_idx % 2 == 1:
    #             image_output = self(batch["i"], mode="image")
    #         total_loss = sum([v for k, v in video_output.items() if "loss" in k]) + sum(
    #             [v for k, v in image_output.items() if "loss" in k]
    #         )
    #     else:
    #         output = self(batch, mode="video")
    #         total_loss = sum([v for k, v in output.items() if "loss" in k])
    #     return total_loss

    # def training_epoch_end(self, outs):
    #     cotrain_utils.epoch_wrapup(self)

    # def validation_step(self, batch, batch_idx):
    #     cotrain_utils.set_task(self)
    #     if "v" in batch and "i" in batch:
    #         video_output = self(batch["v"], mode="video")
    #         image_output = self(batch["i"], mode="image")
    #     else:
    #         output = self(batch, mode="video")

    # def validation_epoch_end(self, outs):
    #     cotrain_utils.epoch_wrapup(self)
    #     self.text_ret = None

    # def configure_optimizers(self):
    #     return cotrain_utils.set_schedule(self)

if __name__ == "__main__":
    import numpy as np
    from torch import utils
    from config import config
    from InternVideo import tokenize
    from Dataset import AnimalKingdomDataset

    device = 'cpu'
    _config = config()
    model = VideoCLIP(_config)
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)
    model.set_text_feats(dataset_train.text_features)
    train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True) # TODO: DEBUG num_workers=4
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False) # TODO: DEBUG num_workers=4

    # test otptimizer
    optimizer = model.configure_optimizers()

    # test forward and train
    for batch_idx, (video_tensor, labels_onehot) in enumerate(train_loader):
        video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
        loss = model.training_step((video_tensor, labels_onehot), batch_idx)
        print(loss)
        break

    for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
        video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
        model.validation_step((video_tensor, labels_onehot), batch_idx)
        break

    # test inference
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
        # video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
        # video_logits = model((video_tensor, labels_onehot))
        # video_logits = video_logits.cpu().detach().numpy()
        # np.save(os.path.join("temp", "video_logits.npy"), video_logits)
        # break
    video_logits = np.load(os.path.join("temp", "video_logits.npy"))
    y = np.where(labels_onehot[0])[0]
    y_pred = np.where(video_logits[0] > 0.05)[0]

    df_action = dataset_train.df_action

    print("Prediction:")
    for idx, prompt in df_action.loc[y, 'prompt'].items():
        print(str(idx).zfill(3) + ":", prompt)
    print("====================================")

    print("Ground Truth:")    
    for idx, prompt in df_action.loc[y_pred, 'prompt'].items():
        print(str(idx).zfill(3) + ":", prompt)
    print("====================================")

