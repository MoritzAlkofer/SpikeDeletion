from utils import all_referential, build_montage, remove_channels, normalize, cut_and_jitter
from make_datamodule import datamodule, get_split_df
import numpy as np
import os
import pickle
import pandas as pd
import torch
from spikenet_model import SpikeNetInstance
from sklearn.metrics import roc_curve, auc
from model import EEGTransformer
import pytorch_lightning as pl
from tqdm import tqdm
import argparse

def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config

def load_model_from_checkpoint(path_model,config,architecture):
   if architecture =='Transformer':
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
   elif architecture =='SpikeNet':
      model = SpikeNetInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=len(config.CHANNELS),map_location=torch.device('cuda'))                        
   return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def save_preds(df,preds,path_model):
   df['preds'] = preds
   df = df[['event_file','preds','total_votes_received','fraction_of_yes','Mode']]
   df.to_csv(path_model+'/pred.csv',index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    args = parser.parse_args()
    return args.path_model

if __name__=='__main__':

   path_model = '../Models/specialized_two_referential_aug'
   architecture ='SpikeNet'
   path_data = '/media/moritz/internal_expansion/Data/Bonobo/cluster_center/' 
   config = get_config(path_model)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')

   montage = build_montage(config.CHANNELS,all_referential)
   cutter = cut_and_jitter(1,0,config.FQ)
      
   n_runs = 5
   results = {'n_keeper':[],'run':[],'AUROC':[]}
   model = load_model_from_checkpoint(path_model,config,architecture)
   for n_keeper in tqdm(range(len(config.CHANNELS)+1)):
      for run in range(n_runs):
         channel_remover = remove_channels(len(config.CHANNELS),N_keeper=n_keeper)
         transforms = [montage,cutter,normalize,channel_remover]
         module = datamodule(transforms=transforms,batch_size=256,echo=False)
         dataloader, df = module.test_dataloader(), get_split_df(module.df,'Test')
         preds = generate_predictions(model,trainer,dataloader)
         labels = df.fraction_of_yes.round(0).astype(int)
         fpr, tpr, thresholds = roc_curve(labels, preds)
         roc_auc = auc(fpr, tpr)
         print(f'n_keeper: {n_keeper}, AUROC: {roc_auc}')
         results['n_keeper'].append(n_keeper)
         results['run'].append(run)
         results['AUROC'].append(roc_auc)
   results = pd.DataFrame(results)
   results.to_csv(os.path.join(path_model,'results_deleted.csv'),index=False)
      
