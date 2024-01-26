from a_train_model import Config
import numpy as np
import os
import pickle
import pandas as pd
from utils import *
from utils import cut_and_jitter, build_montage, SpikeDataset
from torch.utils.data import DataLoader
from model import EEGTransformer
import pytorch_lightning as pl
import argparse

def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config

def init_location_dict():
   # init channel and spike locations
   location_dict = {'frontal':['F3','F4'],
         'parietal':['P3','P4'],
         'occipital':['O1','O2'],
         'temporal':['T3','T4'],
         'central':['C3','C4'],
         'general':['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']}
   return location_dict

class keep_fixed_number_of_fixed_channels():
    # init with total number of channels + channels to be retained
    # use: input, signal, list of channels to be retained
    # output: zero masked signal with *channels from list retained*
    def __init__(self,montage_channels,keeper_channels):
        self.montage_channels = montage_channels
        self.keeper_channels = keeper_channels
      #   print('keeping the following channels: '+keeper_channels)
    def __call__(self,signal):
        keeper_indices = np.array([self.montage_channels.index(channel) for channel in self.keeper_channels])        
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output

def build_channel_deleter(keeper_channels,config):
   channel_deleter = keep_fixed_number_of_fixed_channels(montage_channels=config.CHANNELS,keeper_channels=keeper_channels)
   return channel_deleter

def build_dataloader(df,path_data,storage_channels,transforms,config):
   montage = build_montage(montage_channels=config.CHANNELS,storage_channels=storage_channels)
   windowcutter = cut_and_jitter(windowsize=config.WINDOWSIZE,max_offset=0,Fq=config.FQ)
   # set up dataloaders
   dataset_test = SpikeDataset(df, path_data, windowcutter=windowcutter, montage=montage, transform=transforms)
   test_dataloader = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
   return test_dataloader

def init_dataset_center():
   df = pd.read_csv('../Data/tables/lut_event_23-08-22.csv')
   df = df[df.Mode=='Test'].copy()
   storage_channels = all_referential
   path_data = '../Data/cluster_center'
   return df, storage_channels, path_data

def init_dataset_HashFolder():
   df = pd.read_csv('../Data/tables/locations_HashFolders.csv')
   df['fraction_of_yes']=1
   storage_channels = all_referential
   path_data = '../Data/extracted_localized_hashfolder_processed'
   return df, storage_channels, path_data

def load_model_from_checkpoint(path_model,config):
   model = EEGTransformer.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),
                                          lr=config.LR,
                                          head_dropout=config.HEAD_DROPOUT,
                                          n_channels=len(config.CHANNELS),
                                          n_fft=config.N_FFT,
                                          hop_length=config.HOP_LENGTH)# add this if running on CPU machine
   return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,accelerator='cpu')
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def save_preds(df,preds,path_model):
   df['preds'] = preds
   df = df[['event_file','preds','total_votes_received','fraction_of_yes']]
   df.to_csv(path_model+'/pred.csv',index=False)

if __name__=='__main__':

   path_model = '../Models/generalized'
   

   # df, storage_channels = init_dataset_bonobo()
   df, storage_channels, path_data = init_dataset_HashFolder()
   config = get_config(path_model)
   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   location_dict = init_location_dict()

   results = {'event_file':[],'fraction_of_yes':[],'pred':[],'ChannelLocation':[]}
   for location,keeper_channels in location_dict.items():
      channel_deleter = build_channel_deleter(keeper_channels,config)
      dataloader = build_dataloader(df,path_data,storage_channels,config=config,transforms=channel_deleter)
      preds = generate_predictions(model,trainer,dataloader)
      results['event_file']+=df.event_file.to_list()
      results['fraction_of_yes']+=df.fraction_of_yes.to_list()
      results['pred']+=list(preds)
      results['ChannelLocation']+=[location]*len(df.event_file)

results = pd.DataFrame(results)
results.to_csv(path_model+f'results.csv',index=False)