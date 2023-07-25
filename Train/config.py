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

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for training
    video_sampling = 'rand' # 'rand', 'uniform', 'sequence_rand'
    batch_size = 128
    max_epochs = 100
    lr = 0.0001
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine
    warmup_steps = 10000
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay

    version = None
    data_workers = 12
    functional_test_size = None
    
    # for model
    loss = "BCE" # "BCE", "FOCAL_2", "LDAM", "EQL"
    n_classes = 140
    train_laryers = "vision_proj" # vision or vision_proj

    ckpt_path_imageclip = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt"))
    ckpt_path_videoclip = os.path.abspath(os.path.join(base_dir, "weights", "InternVideo-MM-L-14.ckpt"))
    ckpt_path = None
    num_frames = 8
    clip_evl_dropout = 0.0
    clip_no_pretrain = False
    clip_init_zero = True
    clip_dpr = 0.0
    clip_use_checkpoint = False
    clip_checkpoint_num = [0, 0, 0]

    # slowfast model ()
    # enable the fast stream in the model
    # or only use the slow stream (videoclip)
    slowfast = True 

    # use different sampling settings for fast stream
    # if True, the AnimalKingdomDatasetDiffSampling will be used, 
    # else AnimalKingdomDataset will be used
    diff_sampling_fast = True 

    ## fast stream dataset setting
    num_frames_fast = 32
    video_sampling_fast = 'rand' # 'rand', 'uniform', 'sequence_rand'
    
    ## fast stream model setting
    use_image_clip_fast = True # use pretrained (image clip / african) as image encoder in fast stream
    ckpt_path_fast = os.path.abspath(os.path.join(base_dir, "weights", "ViT-L-14.pt")) # clip_cosine_infoNCE_8_uniform_augmix_map_epoch124.ckpt
    transformer_width_fast = 768
    transformer_layers_fast = 6
    transformer_heads_fast = 12

    ## fast stream preprocess setting
    # also means two different source for two streams of the model
    # affetct Model.py: if True, the dataset output should be video feature embedding, else output should be video frame
    # affetct train.py: 
    enable_preprocess_fast = True 
    preprocess_dir_fast = os.path.join(os.path.dirname(__file__), 'preprocess', "video_feats") 



    