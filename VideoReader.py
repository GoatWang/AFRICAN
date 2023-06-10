import random
import decord
import numpy as np
from decord import cpu

def read_frames_decord(video_path, num_frames=8, mode='train', fix_start=None):
    if mode in ['train']:
        sample = 'rand'
    else:
        sample = 'uniform'
    # video_reader = decord.VideoReader(video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    # video_reader = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames.permute(0, 3, 1, 2).cpu()
    return frames, frame_idxs, vlen

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
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
    from pathlib import Path
    from datetime import datetime
    from data_location import data_dir
    temp_dir = os.path.join('temp', 'VideoReader')
    video_fp = os.path.join(data_dir, "dataset/image/AABCQPTK.mp4")
    save_dir = os.path.join(temp_dir, os.path.basename(video_fp).split(".")[0] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    frames, frame_idxs, vlen = read_frames_decord(video_fp, num_frames=8, fix_start=True)
    for frame, frame_idx in zip(frames, frame_idxs):
        cv2.imwrite(os.path.join(save_dir, f"{frame_idx}.jpg"), frame.numpy().transpose(1, 2, 0))
