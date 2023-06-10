import os
from sacred import Experiment
ex = Experiment("AnimalKingdomCLIP")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "AnimalKingdomCLIP"
    version = "0.0.1"

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for training
    max_epochs = 10
    lr = 0.0001
    optimizer = "adamw" # adam or adamw
    
    # for model
    n_classes = 140
    clip = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    load_path = os.path.abspath(os.path.join(base_dir, "weights", "InternVideo", "models", "InternVideo-MM-L-14.ckpt"))
    num_frames = 8
    clip_evl_dropout = 0.0
    clip_no_pretrain = False
    clip_init_zero = True
    clip_dpr = 0.0
    clip_dpr = 0.0
    clip_dpr = 0.0
    clip_use_checkpoint = False
    clip_checkpoint_num = [0, 0, 0]