from local_utils import all_referential, build_montage, remove_channels, normalize, cut_and_jitter, Config, remove_channels
from make_datamodule import datamodule
import wandb
import torch
from model import EEGTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from make_datamodule import datamodule, datamoduleLocal

import argparse
from local_utils import all_referential, six_referential, two_referential, all_bipolar, six_bipolar

def get_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fq,random_delete=False):
    montage = build_montage(montage_channels,storage_channels)
    cutter = cut_and_jitter(windowsize,windowjitter,Fq)
    if not random_delete:
        transforms = [montage,cutter,normalize] 
        print('keeping all channels')
        return transforms
    else: 
        deleter = remove_channels(montage_channels,'random')
        transforms = [montage,cutter,normalize,deleter] 
        print('deleting random channels')
        return transforms


def get_model():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--model_name')
    model_name = parser.parse_args().model_name
    if model_name == 'specialized_all_referential':
        channels = all_referential
        random_delete = False
        split = 'representative'
    if model_name == 'specialized_six_referential':
        channels = six_referential
        random_delete = False
        split = 'representative'
    if model_name == 'specialized_two_referential':
        channels = two_referential
        random_delete = False
        split = 'representative'
    if model_name == 'specialized_six_bipolar':
        channels = six_bipolar
        random_delete = False
        split = 'representative'
    if model_name == 'specialized_all_bipolar':
        channels = all_bipolar
        random_delete = False
        split = 'representative'
    if model_name == 'generalized_referential':
        channels = all_referential
        random_delete = True
        split = 'representative'
    if model_name == 'generalized_all_ref_loc':
        channels = all_referential
        random_delete = True
        split = 'localized'

    return model_name, channels, random_delete, split


if __name__ == '__main__':
    model_name, channels, random_delete, split = get_model()
    config = Config()
    path_model = '../Models/'+model_name
    config.save_config(path_model)

    
    transforms = get_transforms(config.CHANNELS,all_referential,config.WINDOWSIZE,config.WINDOWJITTER,config.FQ,random_delete)
    if split =='representative':
        module = datamodule(transforms=transforms,batch_size=config.BATCH_SIZE)
    elif split =='localized':
        module = datamoduleLocal(transforms=transforms,batch_size=config.BATCH_SIZE)
    

    # create a logger
    wandb.init(name = path_model.split('/')[-1],
            dir='.logging')
    wandb_logger = WandbLogger(project='SpikeTransformer') 

    torch.set_float32_matmul_precision('high')
    # create callbacks with early stopping and model checkpoint (saves the best model)
    callbacks = [EarlyStopping(monitor='val_loss',patience=20),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    # create trainer, use fast dev run to test the code
    trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, callbacks=callbacks, devices=1)
    # build and train the model
    model = EEGTransformer(lr=config.LR, head_dropout=config.HEAD_DROPOUT, n_channels=len(config.CHANNELS), n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, depth= config.DEPTH, heads=config.HEADS, emb_size=config.EMB_SIZE, weight_decay=config.WEIGHT_DECAY)
    trainer.fit(model,datamodule=module)
