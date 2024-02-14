from tqdm import tqdm
from local_utils import Config
import numpy as np
import os
import pickle
import pandas as pd
#from utils import *
from local_utils import cut_and_jitter, build_montage, SpikeDataset
from local_utils import all_referential
from torch.utils.data import DataLoader
from model import EEGTransformer
import torch
from sklearn.metrics import roc_curve, auc
import pytorch_lightning as pl

def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config

def apply_filters(df):
    pos = df[(df['fraction_of_yes'] >= 7/8) & (df.total_votes_received >=8) &(df.Mode=='Test')]
    neg = df[(df.fraction_of_yes==0)&(df.Mode=='Test')]
    N = min(len(pos),len(neg))
    df = pd.concat([pos[:N],neg[:N]])    
    print(N)
    return df

class keep_fixed_number_of_random_channels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,montage_channels,n_keeper_channels):
        self.n_channels = len(montage_channels)
        self.n_keeper_channels = n_keeper_channels
    def __call__(self,signal):
        # choose n random keeper_channels
        keeper_indices = np.random.choice(self.n_channels, self.n_keeper_channels, replace=False)
        # build output
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output

def build_dataloader(df,path_data,storage_channels,transforms,config):
   montage = build_montage(montage_channels=config.CHANNELS,storage_channels=storage_channels,echo=False)
   windowcutter = cut_and_jitter(windowsize=config.WINDOWSIZE,max_offset=0,Fq=config.FQ)
   # set up dataloaders
   dataset_test = SpikeDataset(df, path_data, windowcutter=windowcutter, montage=montage, transform=transforms,echo=False)
   test_dataloader = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
   return test_dataloader

def init_dataset_center():
   df = pd.read_csv('../Data/tables/lut_event_23-08-22.csv')
   df = df[df.Mode=='Test'].copy()
   storage_channels = all_referential
   path_data =    path_data = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_center/' 
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
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,accelerator='gpu',devices=[0])
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

   path_model = '../Models/generalized_rep'
   
   df, storage_channels, path_data = init_dataset_center()
   df = apply_filters(df)
   config = get_config(path_model)
   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   
   labels = df.fraction_of_yes.round(0).astype(int)
   results = {'n_keeper':[],'run':[],'AUROC':[]}
   for n_keeper in tqdm(range(20)):
      for run in tqdm(range(5),leave =False):
         channel_deleter = keep_fixed_number_of_random_channels(config.CHANNELS,n_keeper)
         dataloader = build_dataloader(df,path_data,storage_channels,config=config,transforms=channel_deleter)
         preds = generate_predictions(model,trainer,dataloader)
         print(len(preds),len(labels))
         fpr, tpr, thresholds = roc_curve(labels, preds)
         roc_auc = auc(fpr, tpr)

         results['n_keeper'].append(n_keeper)
         results['run'].append(run)
         results['AUROC'].append(roc_auc)
   results = pd.DataFrame(results)
   results.to_csv(os.path.join(path_model,'results_channel_deletion.csv'),index=False)