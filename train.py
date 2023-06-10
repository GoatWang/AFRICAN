import numpy as np
from torch import utils
from config import config
from Model import VideoCLIP
import pytorch_lightning as pl
from data_location import data_dir
from Dataset import AnimalKingdomDataset

device = 'cpu'
model = VideoCLIP(config)
dataset_train = AnimalKingdomDataset(config['data_dir_train'], split="train")
dataset_valid = AnimalKingdomDataset(config['data_dir_valid'], split="val")
dataset_train.produce_prompt_embedding(model.clip)
dataset_valid.produce_prompt_embedding(model.clip)
model.set_text_feats(dataset_train.text_features)

train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True) # TODO: DEBUG num_workers=4
valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False) # TODO: DEBUG num_workers=4

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config["model_dir"], save_top_k=3, every_n_train_steps=1, verbose=True)
# monitor="contrastive/train/loss", mode="min", save_last=_config["save_last"], ,
# lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
summary_callback = pl.callbacks.ModelSummary(max_depth=1)

logger = pl.loggers.CSVLogger(config["log_dir"], name=config['name'], version=config['version'])
trainer = pl.Trainer(max_epochs=config['max_epochs'], logger=logger, callbacks=[checkpoint_callback, summary_callback])
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


# optimizer = model.configure_optimizers()
# for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
#     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
#     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()