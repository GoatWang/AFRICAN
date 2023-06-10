config = {
    "name": "VideoCLIP",
    "version": "0.0.1",
    "log_dir": "logs",
    "max_epochs": 10,
    "lr": 0.0001,
    "optimizer": "adamw", # adam, adamw
    "n_classes": 140,

    "model_dir": "ckpts",
    
    "clip": "./weights/ViT-L-14.pt",
    "num_frames": 8,
    "clip_evl_dropout": 0.0,
    "clip_no_pretrain": False,
    "clip_init_zero": True,
    "clip_dpr": 0.0,
    "clip_dpr": 0.0,
    "clip_dpr": 0.0,
    "clip_use_checkpoint": False,
    "clip_checkpoint_num": [0, 0, 0],
    "load_path": "weights/InternVideo/models/InternVideo-MM-L-14.ckpt",
}