import torch
import numpy as np

class MyCollate:
    def __init__(self, config, image_encoder_ic, image_encoder_af):
        self.device = config['device']
        self.preprocess_batch_size = config['preprocess_batch_size']
        self.transformer_width_ic = config['transformer_width_ic']
        self.transformer_width_af = config['transformer_width_af']
        self.image_encoder_ic = image_encoder_ic
        self.image_encoder_af = image_encoder_af
        self.enable_image_clip = config['enable_image_clip']
        self.enable_african = config['enable_african']

    def batch_inference(self, image_encoder, video_tensors_raw, transformer_width):
        B, F, C, H, W = video_tensors_raw.shape
        video_tensors = video_tensors_raw.reshape(B*F, C, H, W)
        feats_tensors = torch.zeros(B*F, transformer_width)
        n_iters = int(np.ceil(feats_tensors.shape[0] / self.preprocess_batch_size))
        for idx in range(n_iters):
            st, end = idx*self.preprocess_batch_size, (idx+1)*self.preprocess_batch_size
            feats_tensors[st:end] = image_encoder(video_tensors[st:end])
        feats_tensors = feats_tensors.reshape(B, F, transformer_width)

        # TODO: debug
        debug = True
        if debug:
            feats_tensors_debug = torch.zeros(B, F, transformer_width)
            for b in range(B):
                feats_tensors_debug[b] = image_encoder(video_tensors_raw[b])
            assert torch.all(torch.isclose(feats_tensors, feats_tensors_debug, rtol=1e-03)), "inference error"

        return feats_tensors

    def __call__(self, batch):
        # frames_tensor_vc, frames_tensor_fast, labels_onehot, index
        frames_tensor_vc, frames_tensor_fast, labels_onehot, index = list(zip(*batch))
        frames_tensor_vc = torch.stack(frames_tensor_vc, dim=0)
        frames_tensor_fast = torch.stack(frames_tensor_fast, dim=0).to(self.device)
        labels_onehot = torch.stack(labels_onehot, dim=0)
        index = list(index)

        feats_tensor_ic = torch.zeros(frames_tensor_fast.shape[0], frames_tensor_fast.shape[1], self.transformer_width_ic)
        if self.enable_image_clip:
            feats_tensor_ic = self.batch_inference(self.image_encoder_ic, frames_tensor_fast.clone(), self.transformer_width_ic)

        feats_tensor_af = torch.zeros(frames_tensor_fast.shape[0], frames_tensor_fast.shape[1], self.transformer_width_af)
        if self.enable_african:
            feats_tensor_af = self.batch_inference(self.image_encoder_af, frames_tensor_fast.clone(), self.transformer_width_af)

        return frames_tensor_vc, feats_tensor_ic, feats_tensor_af, labels_onehot, index
