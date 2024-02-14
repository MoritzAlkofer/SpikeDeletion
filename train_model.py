from local_utils import  build_montage, cut_and_jitter, Config
from local_utils import get_transforms
from local_utils import all_bipolar, all_referential,six_referential, two_referential, four_bipolar
from make_splits import datamodule
import numpy as np
import argparse
import os
import torch
import pytorch_lightning as pl
from model import EEGTransformer
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
            'four_bipolar':four_bipolar}

def init_model(model):
    model = EEGTransformer(lr=config.LR,
                    head_dropout=config.HEAD_DROPOUT,
                    n_channels=len(config.CHANNELS),
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    depth= config.DEPTH,
                    heads=config.HEADS,
                    emb_size=config.EMB_SIZE,
                    weight_decay=config.WEIGHT_DECAY,
                    )
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
    transforms = get_transforms(config.CHANNELS,all_referential,10,2.5,128,random_delete)
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
   