import copy
import torch
from torch import utils
from Model import VideoCLIP
from config import ex, config
import pytorch_lightning as pl
from Dataset import AnimalKingdomDataset
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    model = VideoCLIP(_config)
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    dataset_train.produce_prompt_embedding(model.clip)
    dataset_valid.produce_prompt_embedding(model.clip)
    model.set_text_feats(dataset_train.text_features)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"]) # bugs on MACOS
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"]) # bugs on MACOS

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=_config["model_dir"], verbose=True) # , save_top_k=3, every_n_train_steps=1
    # monitor="contrastive/train/loss", mode="min", save_last=_config["save_last"], ,
    # lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)

    logger = pl.loggers.CSVLogger(_config["log_dir"], name=_config['name'], version=_config['version'])
    trainer = pl.Trainer(max_epochs=_config['max_epochs'], 
                         logger=logger, 
                         callbacks=[checkpoint_callback, summary_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


    # optimizer = model.configure_optimizers()
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()