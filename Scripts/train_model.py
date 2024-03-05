
import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance, SpikeNetInstance
from local_utils import BipolarMontage,ReferentialMontage, WindowCutter, KeepRandomChannels, RandomScaler, ChannelFlipper
from local_utils import all_referential,all_bipolar, six_bipolar, six_referential, two_referential
from local_utils import datamoduleRep, datamoduleClemson, datamoduleHash, datamoduleLocal

import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def init_montage(storage_channels,montage_channels):
    if '-' in montage_channels[0]:
        montage = BipolarMontage(storage_channels,montage_channels)
    else:
        montage = ReferentialMontage(storage_channels,montage_channels)
    return montage

def init_transforms(channel_drop,storage_channels,montage_channels,windowsize,windowjitter,Fs,scale_percent):
    montage = init_montage(storage_channels,montage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    scaler = RandomScaler(scale_percent=scale_percent)
    flipper = ChannelFlipper(channels=montage_channels,p=0.5)

    transforms = [montage,cutter,scaler,flipper]

    if channel_drop:
        dropper = KeepRandomChannels(len(montage_channels))
        transforms+=[dropper]
    return transforms

def init_callbacks(path_model):
    callbacks = [EarlyStopping(monitor='val_loss',patience=20),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

def get_model(architecture,n_channels,lr,weight_decay):
    if architecture == 'ResNet':
        model = ResNetInstance(n_channels=n_channels,lr=lr,weight_decay=weight_decay)
    if architecture == 'Transformer':
        model = TransformerInstance(lr=lr,
                                    weight_decay=weight_decay,
                                    n_channels=n_channels)
    if architecture =='SpikeNet':
        print(n_channels)
        model = SpikeNetInstance(n_channels=n_channels,lr=lr,weight_decay=weight_decay)
    print('>>> hardcoded model parameters in get_model <<<')
    return model

def get_datamodule(dataset,transforms,batch_size):
    if dataset =='Rep':
        module = datamoduleRep(transforms=transforms,batch_size=batch_size)
    elif dataset == 'Loc':
        module = datamoduleLocal(transforms,batch_size)
    elif dataset == 'Clemson':
        module = datamoduleClemson(transforms,batch_size)
    elif dataset == 'Hash':
        module = datamoduleHash(transforms,batch_size)
    else: 
        raise ValueError('Please specify dataset correctly! Options are: Rep, Loc, Clemson, Hash')
    return module

def get_montage_channels(montage_channels_name):
    if montage_channels_name=='all_referential':
        return all_referential
    if montage_channels_name=='all_bipolar':
        return all_bipolar
    if montage_channels_name=='six_referential':
        return six_referential
    if montage_channels_name=='six_bipolar':
        return six_bipolar
    if montage_channels_name=='two_referential':
        return two_referential
            

def get_args():
    parser = argparse.ArgumentParser(description='calculate AUC for model')
    parser.add_argument('--path_model')
    parser.add_argument('--architecture',choices =['Transformer','ResNet','SpikeNet'],default='SpikeNet')
    parser.add_argument('--montage_channels',choices=['all_referential','all_bipolar','six_referential','six_bipolar','two_referential','two_bipolar'],default='all_bipolar')
    parser.add_argument('--channel_drop',action='store_true')
    parser.add_argument('--dataset',choices= ['Rep','Loc'],default='Rep')
    
    args = parser.parse_args()
    return args.path_model, args.architecture, args.montage_channels, args.channel_drop, args.dataset

if __name__ == '__main__':
    storage_channels = all_referential

    windowsize = 1
    windowjitter = 0.1
    Fs = 128
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-4
    scale_percent = 0.1

    path_model, architecture, montage_channels_name, channel_drop, dataset = get_args()
    montage_channels = get_montage_channels(montage_channels_name)

    transforms = init_transforms(channel_drop,storage_channels,montage_channels,windowsize,windowjitter,Fs,scale_percent)
    modelinstance = get_model(architecture,len(montage_channels),lr,weight_decay)
    modelinstance.to('cuda')
    module = get_datamodule(dataset,transforms,batch_size)
    callbacks = init_callbacks(path_model=path_model)
    wandb_logger = WandbLogger(project='SpikeTransformer') 
    trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, callbacks=callbacks, devices=1)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(modelinstance,datamodule=module)

