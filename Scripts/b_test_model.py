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

def build_dataloader(df,path_data,storage_channels,config):
   montage = build_montage(montage_channels=config.CHANNELS,storage_channels=storage_channels)
   windowcutter = cut_and_jitter(windowsize=config.WINDOWSIZE,max_offset=0,Fq=config.FQ)
   # set up dataloaders
   dataset_test = SpikeDataset(df, path_data, windowcutter=windowcutter, montage=montage, transform=None)
   test_dataloader = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
   return test_dataloader

def init_dataset():
   df = pd.read_csv('../Data/tables/center_17JAN24.csv')
   df = df[df.Mode=='Test'].copy()
   storage_channels = all_referential
   return df, storage_channels

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

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    args = parser.parse_args()
    return args.path_model

if __name__=='__main__':

   path_model = get_args()
   path_data = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_center/' 

   df, storage_channels = init_dataset()
   config = get_config(path_model)
   dataloader = build_dataloader(df,path_data,storage_channels,config)
   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   preds = generate_predictions(model,trainer,dataloader)
   save_preds(df,preds,path_model)