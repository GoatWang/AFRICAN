import os
from sacred import Experiment
ex = Experiment("AnimalKingdomCLIP")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "AnimalKingdomCLIPVisionProj"
    version = None
    device = 'cuda'
    seed = 2023

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for dataset
    n_classes = 140
    batch_size = 128
    video_sampling = 'rand' # 'rand', 'uniform', 'sequence_rand'

    # for training
    loss = "BCE" # "BCE", "FOCAL_2", "LDAM", "EQL"
    lr = 0.0001
    max_epochs = 100
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine
    warmup_steps = 10000
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay

    data_workers = 12
    functional_test_size = None
    
    # for model
    ckpt_path = None # for resume training

    # fast stream: Video Clip model
    enable_video_clip = True
    train_laryers = "vision_proj" # vision or vision_proj
    ckpt_path_imageclip_vc = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    ckpt_path_videoclip_vc = os.path.abspath(os.path.join(base_dir, "weights", "InternVideo-MM-L-14.ckpt"))
    num_frames = 8
    clip_evl_dropout = 0.0
    clip_no_pretrain = False
    clip_init_zero = True
    clip_dpr = 0.0
    clip_use_checkpoint = False
    clip_checkpoint_num = [0, 0, 0]

    ## image clip
    image_encoder_batch_size = 512
    enable_image_clip = True
    ckpt_path_ic = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    transformer_width_ic = 768
    transformer_layers_ic = 6
    transformer_heads_ic = 12

    ## african
    enable_african = True
    ckpt_path_af = os.path.abspath(os.path.join(base_dir, "weights", "clip_nodecay_infoNCE_8_rand_augmix_000030_epoch30.ckpt"))
    transformer_width_af = 768
    transformer_layers_af = 6
    transformer_heads_af = 12

