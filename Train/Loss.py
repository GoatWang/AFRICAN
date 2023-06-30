#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# class SoftTargetCrossEntropy(nn.Module):
#     """
#     Cross entropy loss with soft target.
#     """

#     def __init__(self, reduction="mean"):
#         """
#         Args:
#             reduction (str): specifies reduction to apply to the output. It can be
#                 "mean" (default) or "none".
#         """
#         super(SoftTargetCrossEntropy, self).__init__()
#         self.reduction = reduction

#     def forward(self, x, y):
#         loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "none":
#             return loss
#         else:
#             raise NotImplementedError

class BCELoss(nn.Module):
    '''
    Function: BCELoss
    Params:
        predictions: input->(batch_size, 1004)
        targets: target->(batch_size, 1004)
    Return:
        bceloss
    '''

    def __init__(self, logits=True, reduction="mean"):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)

        return BCE_loss

class FocalLoss(nn.Module):
    '''
    Function: FocalLoss
    Params:
        alpha: scale factor, default = 1
        gamma: exponential factor, default = 0
    Return:
        focalloss
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/facebookresearch/Detectron
    '''

    def __init__(self, gamma=2, logits=True, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = 1 
        self.gamma = gamma # adjusted from 0 to 5
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        print(self.gamma)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss

class LDAM(nn.Module):
    '''
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    '''

    def __init__(self, class_frequency, device, logits=True, reduction="mean", max_m=0.5, s=30, step_epoch=80):
        super(LDAM, self).__init__()
        self.class_frequency = class_frequency
        self.reduction = reduction
        self.logits = logits
        self.device = device

        m_list = 1.0 / np.sqrt(np.sqrt(self.class_frequency))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        self.s = s
        self.step_epoch = step_epoch
        self.weight = None

    # def reset_epoch(self, epoch):
    #     idx = epoch // self.step_epoch
    #     betas = [0, 0.9999]
    #     effective_num = 1.0 - np.power(betas[idx], self.class_frequency)
    #     per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.class_frequency)
    #     self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, inputs, targets):
        targets = targets.to(torch.float32)
        batch_m = torch.matmul(self.m_list[None, :], targets.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        inputs_m = inputs - batch_m

        output = torch.where(targets.type(torch.bool), inputs_m, inputs) # TODO: should be tested
        if self.logits:
            loss = F.binary_cross_entropy_with_logits(self.s * output, targets, reduction=self.reduction,
                                                    weight=self.weight)
        else:
            loss = F.binary_cross_entropy(self.s * output, targets, reduction=self.reduction, weight=self.weight)
        return loss


class EQL(nn.Module):
    '''
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/tztztztztz/eql.detectron2
    '''

    def __init__(self, class_frequency, device, logits=True, reduction="mean", max_tail_num=100, gamma=1.76 * 1e-3):
        super(EQL, self).__init__()
        self.reduction = reduction
        self.logits = logits
        self.device = device

        max_tail_num = max_tail_num
        self.gamma = gamma

        self.tail_flag = [False] * len(class_frequency)
        for i in range(len(self.tail_flag)):
            if class_frequency[i] <= max_tail_num:
                self.tail_flag[i] = True

    def threshold_func(self):
        weight = self.inputs.new_zeros(self.n_c)
        weight[self.tail_flag] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def beta_func(self):
        rand = torch.rand((self.n_i, self.n_c)).to(self.device)
        rand[rand < 1 - self.gamma] = 0
        rand[rand >= 1 - self.gamma] = 1
        return rand

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.n_i, self.n_c = self.inputs.size()

        eql_w = 1 - self.beta_func() * self.threshold_func() * (1 - targets)
        if self.logits:
            loss = F.binary_cross_entropy_with_logits(self.inputs, targets, reduction=self.reduction, weight=eql_w)
        else:
            loss = F.binary_cross_entropy(self.inputs, targets, reduction=self.reduction, weight=eql_w)
        return loss

_LOSSES = {
    # "cross_entropy": nn.CrossEntropyLoss,
    # "bce": nn.BCELoss,
    # "bce_logit": nn.BCEWithLogitsLoss,
    # "soft_cross_entropy": SoftTargetCrossEntropy,

    "BCE": BCELoss,
    "FOCAL": FocalLoss,
    "LDAM": LDAM,
    "EQL": EQL,
}

def get_loss_func(loss_name, class_frequency, device):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    loss_name = loss_name.split("_")[0].upper()
    if "BCE" in loss_name:
        return BCELoss()
    elif "FOCAL" in loss_name:
        gamma = 2
        parameters = loss_name.split("_")[1:]
        if len(parameters) > 0:
            gamma = int(parameters[0])
        return FocalLoss(gamma=gamma)
    elif "LDAM" in loss_name:
        return LDAM(class_frequency, device)
    elif "EQL" in loss_name:
        return EQL(class_frequency, device)
    else:
        raise NotImplementedError("Loss {} is not supported".format(loss_name))

if __name__ == "__main__":
    def test_focal_loss():
        focal_loss = FocalLoss(gamma=2)

        # Scenario 1: low loss case
        # We take logits (model's raw outputs) to be very close to targets
        logits1 = torch.tensor([5.0, -5.0, 5.0, -5.0])  # assuming these as model's raw outputs
        targets1 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # ground truth labels
        loss1 = focal_loss(logits1, targets1)

        # Scenario 2: high loss case
        # We take logits to be very far from targets
        logits2 = torch.tensor([-5.0, 5.0, -5.0, 5.0])  # assuming these as model's raw outputs
        targets2 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # ground truth labels
        loss2 = focal_loss(logits2, targets2)

        print('Loss in low loss case:', loss1.item())
        print('Loss in high loss case:', loss2.item())

        assert loss2 > loss1, "Focal loss in the high loss case should be larger than in the low loss case"

    test_focal_loss()







    # import numpy as np
    # import pandas as pd
    # df = pd.read_excel("/Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/annotation/df_action.xlsx")
    # class_frequency = list(map(float, df["count"].tolist()))

    # loss_result = []
    # for loss_name in _LOSSES.keys():
    #     loss_func = get_loss_func(loss_name)(class_frequency, 'cpu', logits=True)        

    #     ZERO_LOGINVERSE = -13.815509557935018
    #     ONE_LOGINVERSE = 13.815509557935018
    #     EPS = 10**-6

    #     input_zeros = torch.ones((10, 140)) * ZERO_LOGINVERSE
    #     target_zeros = torch.zeros((10, 140))
    #     loss_zeros = loss_func(input_zeros, target_zeros)

    #     input_hfp = input_zeros.clone()
    #     idxs_tn = torch.where((input_hfp < (ZERO_LOGINVERSE + EPS)) & (target_zeros < 0.5))
    #     idxs_tn_cand_idxs = np.random.choice(range(len(idxs_tn[0])), int(0.3*len(idxs_tn[0])), replace=False)
    #     # idxs_tn_cand_idxs.sort()
    #     idxs_tn_cands = [idx_tn[idxs_tn_cand_idxs] for idx_tn in idxs_tn]
    #     input_hfp[idxs_tn_cands] = ONE_LOGINVERSE * 0.5
    #     loss_hfp_half = loss_func(input_hfp, target_zeros)

    #     input_hfp = input_zeros.clone()
    #     idxs_tn = torch.where((input_hfp < (ZERO_LOGINVERSE + EPS)) & (target_zeros < 0.5))
    #     idxs_tn_cand_idxs = np.random.choice(range(len(idxs_tn[0])), int(0.3*len(idxs_tn[0])), replace=False)
    #     # idxs_tn_cand_idxs.sort()
    #     idxs_tn_cands = [idx_tn[idxs_tn_cand_idxs] for idx_tn in idxs_tn]
    #     input_hfp[idxs_tn_cands] = ONE_LOGINVERSE
    #     loss_hfp = loss_func(input_hfp, target_zeros)



    #     input_ones = torch.ones((10, 140)) * ONE_LOGINVERSE
    #     target_ones = torch.ones((10, 140))
    #     loss_ones = loss_func(input_ones, target_ones)

    #     input_hfn = input_ones.clone()
    #     idxs_tp = torch.where((input_hfn > (ONE_LOGINVERSE - EPS)) & (target_ones > 0.5))
    #     idxs_tp_cand_idxs = np.random.choice(range(len(idxs_tp[0])), int(0.3*len(idxs_tp[0])), replace=False)
    #     # idxs_tp_cand_idxs.sort()
    #     idxs_tp_cands = [idx_tp[idxs_tp_cand_idxs] for idx_tp in idxs_tp]
    #     input_hfn[idxs_tp_cands] = ZERO_LOGINVERSE * 0.5
    #     loss_hfn_half = loss_func(input_hfn, target_ones)
    #     print(input_hfn)

    #     input_hfn = input_ones.clone()
    #     idxs_tp = torch.where((input_hfn > (ONE_LOGINVERSE - EPS)) & (target_ones > 0.5))
    #     idxs_tp_cand_idxs = np.random.choice(range(len(idxs_tp[0])), int(0.3*len(idxs_tp[0])), replace=False)
    #     # idxs_tp_cand_idxs.sort()
    #     idxs_tp_cands = [idx_tp[idxs_tp_cand_idxs] for idx_tp in idxs_tp]
    #     input_hfn[idxs_tp_cands] = ZERO_LOGINVERSE
    #     loss_hfn = loss_func(input_hfn, target_ones)
    #     print(input_hfn)

    #     loss_result.append({
    #         'loss_zeros': loss_zeros.item(),
    #         'loss_hfp_half': loss_hfp_half.item(),
    #         'loss_hfp': loss_hfp.item(),
    #         'loss_ones': loss_ones.item(),
    #         'loss_hfn_half': loss_hfn_half.item(),
    #         'loss_hfn': loss_hfn.item()
    #     })
    
    # # columns = 'loss_ori', 'loss_hfp', 'loss_hfn'
    # df_loss = pd.DataFrame(loss_result)# , columns=columns
    # df_loss.index = _LOSSES.keys()
    # print(df_loss)