import pytorch_lightning as pl
import numpy as np
import torch
import os
import json
from .models.instances import SpikeNetInstance
def get_config(path_model):
    with open(os.path.join(path_model,'config.json'), 'r') as fp:
        config = json.load(fp)
    return config

def load_model_from_checkpoint(path_model,config):
   if config['ARCHITECTURE'] =='SpikeNet':
      model = SpikeNetInstance.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),n_channels=len(config['CHANNELS']),map_location=torch.device('cuda'))                        
   return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,accelerator='cpu')
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds
