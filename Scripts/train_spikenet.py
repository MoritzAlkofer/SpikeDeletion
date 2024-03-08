from utils import  build_montage, cut_and_jitter, Config
from utils import get_transforms
from utils import all_bipolar, all_referential,six_referential, two_referential, six_bipolar
from make_datamodule import datamodule
import numpy as np
import argparse
import os
import torch
import pytorch_lightning as pl
from spikenet_model import SpikeNetInstance
from model import EEGTransformer
from spikenet_model import SpikeNet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--model_name',default = 'model')
    parser.add_argument('--channels',default='two_referential', choices=['all_referential','six_referential','two_referential','all_bipolar','four_bipolar'])
    parser.add_argument('--random_delete',default=False, action= 'store_true')
    parser.add_argument('--split',default='representative', choices=['representative','localized'])
    args = parser.parse_args()
    return args.model_name, args.channels, args.split, args.random_delete

channel_dict = {'all_referential': all_referential,
            'six_referential': six_referential,
            'two_referential': two_referential,
            'all_bipolar': all_bipolar,
            'six_bipolar':six_bipolar}

def init_model(model):
    model = SpikeNetInstance(n_channels=19)
    return model

def init_callbacks(path_model, patience=20):
    callbacks = [EarlyStopping(monitor='val_loss',patience=patience),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

if __name__ == '__main__':
    model_name, channels, split, random_delete = get_args()
    path_model = f'../Models/{model_name}'
    config = Config()
    config.CHANNELS = channel_dict[channels]
    config.save_config(path_model)
    transforms = get_transforms(config.CHANNELS,all_referential,1,0.1,128,random_delete)
    module = datamodule(transforms,config.BATCH_SIZE)

    torch.set_float32_matmul_precision('high')
    callbacks = init_callbacks(path_model)
    trainer = pl.Trainer(max_epochs=300,
                        logger=WandbLogger(project='SpikeDeletion'),
                        callbacks=callbacks,
                        devices=1,
                        fast_dev_run=False)
    # train the model
    model = init_model(config)
    trainer.fit(model,module)
   