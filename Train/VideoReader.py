import torch 
import random
import decord
import numpy as np
from decord import cpu

def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    # video_reader = decord.VideoReader(video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    # video_reader = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    if sample in ['rand', 'uniform']:
        frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    elif sample == ("sequence_rand"):
        step_size = np.random.randint(1, 6)
        frame_idxs = sample_frames_seq(num_frames, vlen, step_size)
    elif sample.startswith("sequence"):
        assert len(sample.split("_")) == 2, "sequence should be like sequence_1, the number represents the step_size"
        step_size = int(sample.split("_")[1])
        frame_idxs = sample_frames_seq(num_frames, vlen, step_size)

    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames.permute(0, 3, 1, 2).cpu()
    if frames.shape[0] < num_frames:
        pad_n_frames = num_frames - frames.shape[0]
        pad_frames = torch.stack([torch.zeros_like(frames[0])] * pad_n_frames)
        frames = torch.cat([frames, pad_frames], dim=0)
    return frames, frame_idxs, vlen

def sample_frames_seq(num_frames, vlen, step_size):
    num_frames_steps = num_frames * step_size
    if vlen > num_frames_steps:
        st_idx = np.random.choice(list(range(vlen - num_frames_steps)))
        end_idx = st_idx + num_frames_steps
    elif vlen == num_frames_steps:
        st_idx = 0
        end_idx = st_idx + num_frames_steps
    else:
        st_idx = 0
        end_idx = vlen
    frame_idxs = range(st_idx, end_idx)[::step_size]
    return frame_idxs

# def read_frames_cv2(video_path, num_frames, mode='train', fix_start=None):
#     if mode in ['train']:
#         sample = 'rand'
#     else:
#         sample = 'uniform'

#     # print(video_path)
#     cap = cv2.VideoCapture(video_path)
#     assert (cap.isOpened())
#     # for decord
#     # cap.set(3, 256)
#     # cap.set(4, 256)
#     vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # get indexes of sampled frames
#     frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
#     frames = []
#     success_idxs = []
#     for index in frame_idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = torch.from_numpy(frame).byte()
#             # # (H x W x C) to (C x H x W)
#             frame = frame.permute(2, 0, 1)
#             # frame = Image.fromarray(frame)
#             frames.append(frame)
#             success_idxs.append(index)
#         else:
#             pass
#             # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
#     frames = torch.stack(frames) # .float() / 255

#     if frames.shape[0] < num_frames:
#         pad_n_frames = num_frames - frames.shape[0]
#         pad_frames = torch.stack([torch.zeros_like(frames[0])] * pad_n_frames)
#         frames = torch.cat([frames, pad_frames], dim=0)

#     # return frames tensor
#     # convert cv to PIL
#     # img = Image.fromarray(imgs[0])
#     # print(frames.size())
#     cap.release()
#     return frames, success_idxs, vlen

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        st = interv
        end = intervals[idx + 1] - 1
        end = st + 1 if st == end else end
        ranges.append((st, end))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs

if __name__ == "__main__":
    import os
    import cv2
    from config import config
    from pathlib import Path
    from datetime import datetime
    _config = config()
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp', 'VideoReader')
    # video_fp = os.path.join(_config['data_dir'], "dataset/video/AABCQPTK.mp4")
    video_fp = os.path.join(_config['data_dir'], "dataset/video/AAOZGQFB.mp4")
    # video_fp = os.path.join(_config['data_dir'], "dataset/video/ZAKHHVKA.mp4")

    save_dir = os.path.join(temp_dir, os.path.basename(video_fp).split(".")[0] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    frames, frame_idxs, vlen = read_frames_decord(video_fp, num_frames=_config['num_frames'], sample=_config['video_sampling'], fix_start=True)
    for frame, frame_idx in zip(frames, frame_idxs):
        cv2.imwrite(os.path.join(save_dir, f"{frame_idx:04d}.jpg"), frame.numpy().transpose(1, 2, 0))
