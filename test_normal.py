import json
from utils import SpikeNetInstance, TransformerInstance
from utils import DatamoduleRep, DatamoduleClemson
from utils import Montage,Cutter, normalize
from utils import all_referential
from utils import get_config, load_model_from_checkpoint
import numpy as np
import os
import pandas as pd
import torch

import pytorch_lightning as pl
import argparse

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def save_preds(path_model,dataset,preds,labels):
   df = pd.DataFrame({'pred':preds,'label':labels})
   df.to_csv(path_model+f'/pred_{dataset}.csv',index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    return args.path_model, args.dataset

def get_transforms(storage_channels,config):
    montage = Montage(montage_channels = config['CHANNELS'],storage_channels=storage_channels)
    cutter = Cutter(config['WINDOWSIZE'],config['WINDOWJITTER'],config['FS'])

    transforms = [montage,cutter,normalize]
    print('keeping all channels')
    return transforms

if __name__=='__main__':

   path_model, dataset = get_args()
   config = get_config(path_model)

   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   transforms = get_transforms(all_referential,config)
   torch.set_float32_matmul_precision('high')
   if dataset == 'Rep':
      module = DatamoduleRep(batch_size=config['BATCH_SIZE'],transforms=transforms)
   if dataset == 'Clemson':
      module = DatamoduleClemson(batch_size=config['BATCH_SIZE'],transforms=transforms)
   preds = generate_predictions(model,trainer,module.test_dataloader())
   labels = module.get_labels('Test')
   save_preds(path_model,dataset,preds,labels)