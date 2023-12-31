import os
import copy
import torch
import torchmetrics
from tqdm import tqdm
from torch import utils
from config import ex, config
import pytorch_lightning as pl
from Model import AfricanSlowfast
from Dataset import AnimalKingdomDataset, AnimalKingdomDatasetSlowFast
torch.manual_seed(0)



@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    pl.seed_everything(_config["seed"])
    Dataset = AnimalKingdomDatasetSlowFast
    dataset_train = Dataset(_config, split="train")
    dataset_valid = Dataset(_config, split="val")
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

    model = AfricanSlowfast(_config).to(_config['device'])
    dataset_train.produce_prompt_embedding(model.video_clip)
    dataset_valid.produce_prompt_embedding(model.video_clip)
    df_action = dataset_train.df_action
    model.set_class_names(df_action['action'].values)
    model.set_text_feats(dataset_train.text_features)
    model.set_loss_func(_config['loss'], df_action['count'].tolist())
    model.set_metrics(df_action[df_action['segment'] == 'head'].index.tolist(), 
                      df_action[df_action['segment'] == 'middle'].index.tolist(), 
                      df_action[df_action['segment'] == 'tail'].index.tolist())

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"])
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"])

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=valid_loader, ckpt_path=_config['ckpt_path'])


# from slowfast.utils.meters import TestMeter
# from slowfast.datasets.charades import Charades

# class TestConfig:
#     def __init__(self):
#         self.NUM_ENSEMBLE_VIEWS = 2
#         self.NUM_SPATIAL_CROPS = 3

# class DataConfig:
#     def __init__(self, path_to_data_dir, path_prefix):
#         self.PATH_TO_DATA_DIR = path_to_data_dir  # Set this value accordingly
#         self.PATH_PREFIX = path_prefix  # Set this value accordingly
#         self.NUM_FRAMES = 8
#         self.SAMPLING_RATE = 8
#         self.TRAIN_JITTER_SCALES = [256, 340]
#         self.TRAIN_CROP_SIZE = 224 # 256
#         self.TEST_CROP_SIZE = 224  # 256
#         self.MEAN = [0.48145466, 0.4578275, 0.40821073] # [0.45, 0.45, 0.45]
#         self.STD = [0.26862954, 0.26130258, 0.27577711] # [0.225, 0.225, 0.225]
#         self.RANDOM_FLIP = True
#         self.INV_UNIFORM_SAMPLE = True
        
# class MultigridConfig:
#     def __init__(self):
#         self.LONG_CYCLE_SAMPLING_RATE = 0
#         self.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]
#         self.DEFAULT_S = 0

# class ModelConfig:
#     def __init__(self):
#         self.NUM_CLASSES = 140

# class Cfg:
#     def __init__(self, path_to_data_dir, path_prefix):
#         self.TEST = TestConfig()
#         self.DATA = DataConfig(path_to_data_dir, path_prefix)
#         self.MULTIGRID = MultigridConfig()
#         self.MODEL = ModelConfig()

# Charades
# @ex.automain
# def main(_config):

#     # load model
#     model = VideoCLIP(_config).to(_config['device'])
#     model.load_ckpt_state_dict(_config['ckpt_path'])
#     model.eval()
#     dataset_train = AnimalKingdomDatasetSlowFast(_config, split="train")
#     dataset_train = Dataset(_config, split="train")
#     _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

#     # load dataset
#     dataset_valid = Dataset(_config, split="val")
#     path_to_data_dir = os.path.join(_config['data_dir'], "annotation")
#     path_prefix = os.path.join(_config['data_dir'], 'dataset', 'image')
#     cfg_charades = Cfg(path_to_data_dir, path_prefix)
#     cfg_charades.DATA.training_test_size = None
#     if _config['training_test_size'] is not None:
#         cfg_charades.DATA.training_test_size = _config['training_test_size']
#     dataset_charades = Charades(cfg_charades, mode='test')
#     # dataset_charades = AnimalKingdomDataset(_config, split="val")
#     loader_charades = utils.data.DataLoader(dataset_charades, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"])

#     # load meter
#     num_cls = cfg_charades.MODEL.NUM_CLASSES
#     num_clips = cfg_charades.TEST.NUM_ENSEMBLE_VIEWS * cfg_charades.TEST.NUM_SPATIAL_CROPS
#     testmeter = TestMeter(len(dataset_charades), num_clips, num_cls, overall_iters=1, multi_label=True)
#     # TODO: modified
#     # testmeter = TestMeter(len(dataset_charades), 1, _config['n_classes'], overall_iters=1, multi_label=True)
#     torch_map = torchmetrics.classification.MultilabelAveragePrecision(num_labels=_config['n_classes'])

#     # evaluate
#     for frames, label, index, _ in tqdm(loader_charades):
#     # for frames, label, index in tqdm(loader_charades):
#         with torch.no_grad():
#             frames, label = frames.to(_config['device']), label
#             video_logits = model((frames, label, index))
#             video_pred = torch.sigmoid(video_logits).detach().cpu()
#             testmeter.update_stats(video_pred, label, index)
#             # TODO: modified
#             torch_map.update(video_pred, label.type(torch.int32))

#     # do stat & print metric
#     testmeter.finalize_metrics()
#     # TODO: modified
#     print("mAP:", torch_map.compute().item())


#     # for testing
#     # dataset_self = AnimalKingdomDataset(_config, split="val")
#     # loader_self = utils.data.DataLoader(dataset_self, batch_size=_config['batch_size'], shuffle=False) # , num_workers=_config["data_workers"]
#     # for batch_idx, (video_tensor, labels_onehot) in enumerate(loader_self):
#     #     video_tensor, labels_onehot = video_tensor.to(_config['device']), labels_onehot.to(_config['device'])
#     #     print("video_tensor.shape", video_tensor.shape)
#     #     print("labels_onehot.shape", labels_onehot.shape)
#     #     break

#     # frames, label, index, _ = charades[0]
#     # print(frames.shape)
#     # print(len(label))
#     # print(index)

#     # for batch_idx, (frames, label, index, _) in enumerate(loader_charades):
#         # for i in range(140):
#         #     print(i, "%.2f, %.2f"%(video_pred[0][i].item(), label[0][i].item()))
#         # break

#     # dataset_charades.get_seq_frames(0)
#     # print("====================================")
#     # dataset_charades.get_seq_frames(1)
#     # print("====================================")
#     # dataset_charades.get_seq_frames(2)
#     # print("====================================")
#     # dataset_charades.get_seq_frames(3)
#     # print("====================================")
#     # dataset_charades.get_seq_frames(4)
#     # print("====================================")
#     # dataset_charades.get_seq_frames(5)
#     # print("====================================")
#     # assert False, "Test"



