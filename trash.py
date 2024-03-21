from local_utils import SpikeNetInstance, TransformerInstance, SpikeNetLargeInstance
from local_utils import DatamoduleRep, DatamoduleLoc
from local_utils import ChannelFlipper, RandomScaler,KeepRandomChannels, Montage, Cutter, normalize
from local_utils import all_bipolar, all_referential, all_average
from local_utils import get_datamodule
import argparse
import os
import numpy as np
import json
from functools import partial
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.signal import resample_poly

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    parser.add_argument('--channels', choices=['all_referential','all_bipolar','all_average'],default='all_referential')
    parser.add_argument('--random_delete', action= 'store_true')
    parser.add_argument('--dataset',default='Rep', choices=['Rep','Loc'])
    parser.add_argument('--architecture',default='SpikeNet')
    args = parser.parse_args()
    return args.path_model, args.channels, args.dataset, args.random_delete, args.architecture

def default_config(channels,windowsize=1,windowjitter=0.1,Fs=128,batch_size=128,
                   lr=1e-4,weight_decay=1e-4,scale_percent=0.1,
                   architecture='SpikeNet',random_delete=False,dataset='Rep'):

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
    if config['ARCHITECTURE'] == 'SpikeNetLarge':
        model = SpikeNetLargeInstance(n_channels = len(config['CHANNELS']))
    elif config['ARCHITECTURE'] =='Transformer':
        config['WINDOWSIZE'] =10
        model = TransformerInstance(lr=config['LR'], n_channels=len(config['CHANNELS']),weight_decay=config['WEIGHT_DECAY'])
    return model

def init_callbacks(path_model, patience=20):
    callbacks = [EarlyStopping(monitor='val_loss',patience=patience),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

def get_transforms(config,storage_channels):

    montage = Montage(montage_channels = config['CHANNELS'],storage_channels=storage_channels)
    cutter = Cutter(config['WINDOWSIZE'],config['WINDOWJITTER'],config['FS'])
    flipper = ChannelFlipper(channels=config['CHANNELS'],p=0.5)
    scaler = RandomScaler(scale_percent=config['SCALE_PERCENT'])
    resample = partial(resample_poly,up=200,down=128,axis=1)
    transforms = [np.nan_to_num,montage,cutter,scaler,flipper,resample]
    
    if config['RANDOM_DELETE']:
        channel_remover = KeepRandomChannels(len(config['CHANNELS']))
        transforms += [channel_remover]
        print('keeping all channels')
    if config['NORMALIZE']:
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
    if channels =='all_referential':
        config = default_config(channels=all_referential,architecture=architecture,random_delete=random_delete)
    elif channels =='all_bipolar':
        config = default_config(channels=all_bipolar,architecture=architecture,random_delete=random_delete)
    elif channels =='all_average':
        config = default_config(channels=all_average,architecture=architecture,random_delete=random_delete
                                )
    if architecture == 'Transformer':
        config['WINDOWSIZE'] = 10
        config['WINDOWJITTER'] = 2.5
        config['FQ']=200
        config['CHANNELS'].remove('Fz')
        config['N_FFT'] = 200
        config['HOP_LENGTH'] = 100
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
    
    path_model = '/media/moritz/Expansion/other_models/BIOT/codes/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt'
    model = TransformerInstance(n_channels=18,emb_size=256,heads=8,depth=4,config['N_FFT'],config['HOP_LENGTH'])
    print(f'>>>{model.model.n_fft}')
    model.model.load_state_dict(torch.load(path_model))
    for block in model.model.transformer.layers.layers[:3]:
        print(11111)
        for layer in block:
            for param in layer.parameters():
                param.requires_grad=False

    module = get_datamodule(config['DATASET'],config['BATCH_SIZE'],transforms)
    data,label = next(iter(module.test_dataloader()))
    print(data.shape)
    trainer.fit(model,module)
   
