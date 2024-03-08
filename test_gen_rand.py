from utils import Montage, Cutter, normalize
from utils import DatamoduleRep
from utils import get_config, load_model_from_checkpoint
from utils import all_referential, KeepNRandomChannels
import numpy as np
import os
import pickle
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
import pytorch_lightning as pl
from tqdm import tqdm
import argparse

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

   path_model = get_args()
   config = get_config(path_model)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')

   montage = Montage(config['CHANNELS'],all_referential)
   cutter = Cutter(config['WINDOWSIZE'],0,config['FS'])
   model = load_model_from_checkpoint(path_model,config)
   transforms = [montage,cutter,normalize]

   n_runs = 3
   results = {'n_keeper':[],'run':[],'AUROC':[]}

   for n_keeper in tqdm(range(len(config['CHANNELS'])+1)):
      for run in range(n_runs):
         channel_remover = KeepNRandomChannels(len(config['CHANNELS']),N_keeper=n_keeper)
         
         module = DatamoduleRep(transforms=transforms+[channel_remover],batch_size=256)
   
         labels = [int(np.round(l,0)) for l in module.get_labels('Test')]
         preds = generate_predictions(model,trainer,module.test_dataloader())
         
         fpr, tpr, thresholds = roc_curve(labels, preds)
         roc_auc = auc(fpr, tpr)
         
         print(f'n_channels {n_keeper}, auc {roc_auc}')
         results['n_keeper'].append(n_keeper), 
         results['run'].append(run), 
         results['AUROC'].append(roc_auc)

   results = pd.DataFrame(results)
   results.to_csv(os.path.join(path_model,'results_deleted.csv'),index=False)
      
