from utils import SpikeNetInstance, TransformerInstance
from utils import DatamoduleRep, DatamoduleLoc
from utils import ChannelFlipper, RandomScaler,KeepRandomChannels, Montage, Cutter, normalize
from utils import all_bipolar, all_referential, all_average
from utils import get_datamodule
import argparse
import os
import numpy as np
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    parser.add_argument('--channels', choices=['all_referential','all_bipolar','all_average'],default='all_referential')
    parser.add_argument('--random_delete',default=False, action= 'store_true')
    parser.add_argument('--dataset',default='Rep', choices=['Rep','Loc'])
    parser.add_argument('--architecture',default='SpikeNet')
    args = parser.parse_args()
    return args.path_model, args.channels, args.dataset, args.random_delete, args.architecture

def default_config(channels,windowsize=1,windowjitter=0.1,Fs=128,batch_size=128,
                   lr=1e-4,weight_decay=1e-4,scale_percent=0.1,
                   architecture='SpikeNet',random_delete=True,dataset='Rep'):

    params = {'CHANNELS': channels,
              'WINDOWSIZE':windowsize,
              'WINDOWJITTER':windowjitter,
              'FS':Fs,
              'BATCH_SIZE':batch_size,
              'LR':lr,
              'WEIGHT_DECAY':weight_decay,
              'SCALE_PERCENT':scale_percent,
              'ARCHITECTURE': architecture,
              'RANDOM_DELETE': random_delete,
              'DATASET':dataset,
              'NORMALIZE':True
            }
    return params
    
def init_model(config):
    if config['ARCHITECTURE'] == 'SpikeNet':
        model = SpikeNetInstance(n_channels = len(config['CHANNELS']))
    elif config['ARCHITECTURE'] =='Transformer':
        model = TransformerInstance(lr=config.LR, head_dropout=config.HEAD_DROPOUT, n_channels=len(config.CHANNELS), n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, depth= config.DEPTH, heads=config.HEADS, emb_size=config.EMB_SIZE, weight_decay=config.WEIGHT_DECAY)
    return model

def init_callbacks(path_model, patience=20):
    callbacks = [EarlyStopping(monitor='val_loss',patience=patience),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

def get_transforms(config,storage_channels):

    montage = Montage(montage_channels = config['CHANNELS'],storage_channels=storage_channels)
    cutter = Cutter(config['WINDOWSIZE'],config['WINDOWJITTER'],config['FS'])
    flipper = ChannelFlipper(channels=config['CHANNELS'],p=0.5)
    scaler = RandomScaler(scale_percent=config['SCALE_PERCENT'])


    transforms = [np.nan_to_num,montage,cutter,scaler,flipper]

    if config['RANDOM_DELETE']:
        channel_remover = KeepRandomChannels(len(config['CHANNELS']))
        transforms += [channel_remover]
        print('keeping all channels')
    transforms+=[normalize]
    print('normalizing')
    return transforms

def save_config(path_model,config):
    if not os.path.isdir(path_model):
        os.mkdir(path_model)
    with open(os.path.join(path_model,'config.json'), 'w') as fp:
        json.dump(config, fp)

if __name__ == '__main__':
    path_model, channels, dataset, random_delete, architecture = get_args()
    print(channels)

    if channels =='all_referential':
        config = default_config(channels=all_referential)
    elif channels =='all_bipolar':
        config = default_config(channels=all_bipolar)
    elif channels =='all_average':
        config = default_config(channels=all_average)
    config['DATASET']=dataset
    save_config(path_model,config)
    transforms = get_transforms(config,storage_channels=all_referential)

    torch.set_float32_matmul_precision('high')
    callbacks = init_callbacks(path_model)
    trainer = pl.Trainer(max_epochs=300,
                        logger=WandbLogger(project='SpikeDeletion',name=path_model.split('/')[-1]),
                        callbacks=callbacks,
                        devices=1,
                        fast_dev_run=False)
    
    model = init_model(config)
    module = get_datamodule(config['DATASET'],config['BATCH_SIZE'],transforms)
    trainer.fit(model,module)
   
