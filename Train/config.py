import os
from sacred import Experiment
ex = Experiment("AnimalKingdomCLIP")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "AnimalKingdomCLIPVisionProj"
    seed = 2023
    device = 'cuda'
    version = None

    # dataset
    n_classes = 140
    num_frames = 32
    data_workers = 12
    functional_test_size = None

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for training
    loss = "BCE" # "BCE", "FOCAL_2", "LDAM", "EQL"
    video_sampling = 'rand' # 'rand', 'uniform', 'sequence_rand'
    batch_size = 128
    max_epochs = 100
    lr = 0.0001
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine
    warmup_steps = 10000
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay
    ckpt_path = None

    ## preprocess setting
    enable_preprocess = True 
    preprocess_dir = os.path.join(os.path.dirname(__file__), 'preprocess', "video_feats") 
    preprocess_batch_size = 512 # since there diff numbers of frame in each video, we need to cat them into a batch for parallel inference

    ## image clip stream setting
    IC_ckpt_path = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    IC_transformer_width = 768
    IC_transformer_layers = 6
    IC_transformer_heads = 12

    # enable african stream
    enable_african = True 
    AF_ckpt_path = os.path.abspath(os.path.join(base_dir, "weights", "clip_cosine_infoNCE_8_uniform_augmix_map_epoch124.ckpt")) # 
    AF_transformer_width = 768
    AF_transformer_layers = 6
    AF_transformer_heads = 12



    