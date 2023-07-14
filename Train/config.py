import os
from sacred import Experiment
ex = Experiment("AnimalKingdomCLIP")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "AnimalKingdomCLIPVisionProj"
    seed = 2023
    device = 'cpu'

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for training
    video_sampling = 'rand' # 'rand', 'uniform', 'sequence_rand'
    batch_size = 32
    max_epochs = 100
    lr = 0.0001
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine
    warmup_steps = 10000
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay

    version = None
    data_workers = 4
    functional_test_size = None
    
    # for model
    loss = "BCE" # "BCE", "FOCAL_2", "LDAM", "EQL"
    n_classes = 140
    train_laryers = "vision_proj" # vision or vision_proj

    clip = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    load_path = os.path.abspath(os.path.join(base_dir, "weights", "InternVideo-MM-L-14.ckpt"))
    animal_kingdom_clip_path = None
    num_frames = 8
    clip_evl_dropout = 0.0
    clip_no_pretrain = False
    clip_init_zero = True
    clip_dpr = 0.0
    clip_use_checkpoint = False
    clip_checkpoint_num = [0, 0, 0]

    # africa
    africa = True
    original_clip_africa = True
    ckpt_path_africa = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    # original_clip_africa = False
    # ckpt_path_africa = os.path.abspath(os.path.join(base_dir, "weights", "clip_cosine_infoNCE_8_uniform_augmix_map_epoch124.ckpt"))
    clip_width_africa = 768
    clip_layers_africa = 4
    clip_heads_africa = 12
    num_frames_africa = 32
    video_sampling_africa = 'uniform' # 'rand', 'uniform', 'sequence_rand'

    