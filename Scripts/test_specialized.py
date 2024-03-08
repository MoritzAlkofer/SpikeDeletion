from utils import remove_channels, build_montage,normalize, cut_and_jitter, Config
from utils import all_referential
from make_datamodule import datamodule, get_split_df, datamoduleClemson
import numpy as np
import os
import pickle
import torch
from spikenet_model import SpikeNetInstance
from model import EEGTransformer
import pytorch_lightning as pl
import argparse

def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config

def load_model_from_checkpoint(path_model,config,n_channels,architecture='SpikeNet'):
   if architecture =='SpikeNet':
      model = SpikeNetInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=n_channels,map_location=torch.device('cuda'))                        
   elif architecture=='Transformer':
      model = EEGTransformer.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),
                                          lr=config.LR,
                                          head_dropout=config.HEAD_DROPOUT,
                                          n_channels=len(config.CHANNELS),
                                          n_fft=config.N_FFT,
                                          hop_length=config.HOP_LENGTH,
                                          heads = config.HEADS,
                                          depth=config.DEPTH,
                                          emb_size = config.EMB_SIZE,
                                          weight_decay = config.WEIGHT_DECAY)
   return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def save_preds(df,preds,path_model,dataset):
   df['preds'] = preds
   df = df[['event_file','preds','total_votes_received','fraction_of_yes','Mode']]
   df.to_csv(path_model+f'/pred_{dataset}.csv',index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    args = parser.parse_args()
    return args.path_model

def get_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fq):
   montage = build_montage(montage_channels,storage_channels)
   cutter = cut_and_jitter(windowsize,windowjitter,Fq)
   transforms = [montage,cutter,normalize]
   print('keeping all channels')
   return transforms

def get_dataset(dataset): 
   if dataset == 'Rep':
      module = datamodule(transforms=transforms,batch_size=256)
      dataloader, df = module.test_dataloader(), get_split_df(module.df,'Test')
   elif dataset == 'Clemson':
         module = datamoduleClemson(transforms=transforms,batch_size=256,echo=False)
         dataloader = module.test_dataloader()
         df = module.df
         df = df.rename(columns={'label':'total_votes_received'})
         df['total_votes_received']=8
   return dataloader, df

if __name__=='__main__':

   path_model = '../Models/specialized_to_referential_aug'
   dataset = 'Rep'
   config = get_config(path_model)
   transforms = get_transforms(config.CHANNELS,all_referential,1,0,128)
   dataloader,df = get_dataset(dataset)
   model = load_model_from_checkpoint(path_model,config,len(config.CHANNELS),architecture=config.ARCHITECTURE)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   preds = generate_predictions(model,trainer,dataloader)
   save_preds(df,preds,path_model,dataset)