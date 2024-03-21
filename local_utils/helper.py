import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from sklearn.utils import resample
import os
import json
from .models.instances import SpikeNetInstance, SpikeNetLargeInstance, TransformerInstance
from .datamodules import DatamoduleClemson, DatamoduleLoc, DatamoduleRep

def get_config(path_model):
    with open(os.path.join(path_model,'config.json'), 'r') as fp:
        config = json.load(fp)
    return config

def load_model_from_checkpoint(path_model,config):
   if config['ARCHITECTURE'] =='SpikeNet':
      model = SpikeNetInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=len(config['CHANNELS']),map_location=torch.device('cuda'))                        
   if config['ARCHITECTURE'] =='SpikeNetLarge':
      model = SpikeNetLargeInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=len(config['CHANNELS']),map_location=torch.device('cuda'))                        
   if config['ARCHITECTURE'] =='Transformer':
         model = TransformerInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=len(config['CHANNELS']),map_location=torch.device('cuda'),n_fft=config['N_FFT'],hop_legth=config['HOP_LENGTH'])                        
   return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,accelerator='cpu')
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def get_datamodule(dataset,batch_size,transforms):
   if dataset == 'Rep':
      module = DatamoduleRep(batch_size=batch_size,transforms=transforms)
   if dataset == 'Loc':
      module = DatamoduleLoc(batch_size=batch_size,transforms=transforms)
   elif dataset == 'Clemson':
      module = DatamoduleClemson(batch_size=batch_size,transforms=transforms)
   return module

def binarize(values,threshold):
    values = np.array(values)
    values[values<threshold] = 0
    values[values!=0] =1
    return list(values)

def calculate_boostraps(preds,labels,n_bootstraps):
    bootstrap = []
    for i in tqdm(range(n_bootstraps)):
        # Resample predictions and true labels
        preds_resampled, labels_resampled = resample(preds, labels)
        # Calculate the metric for the resampled data
        bootstrap.append(roc_auc_score(labels_resampled, preds_resampled))
    return np.array(bootstrap)