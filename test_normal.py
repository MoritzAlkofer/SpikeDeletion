
from utils import two_frontal, two_central, six_referential, all_referential
from utils import init_standard_transforms
from utils import all_referential, two_frontal, two_central
from utils import get_config, load_model_from_checkpoint
from utils import get_datamodule

import numpy as np
import os
import pandas as pd
import torch
import pytorch_lightning as pl
import argparse
from utils import KeepFixedChannels

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def save_preds(path_model,savename,preds,labels,event_files):
   df = pd.DataFrame({'event_file':event_files,'pred':preds,'label':labels})
   df.to_csv(os.path.join(path_model,savename),index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    parser.add_argument('--dataset')
    parser.add_argument('--keeper_channels',default='all_referential')
    args = parser.parse_args()
    return args.path_model, args.dataset, args.keeper_channels

def remove_channels(montage_channels,keeper_channels):
   channel_remover = KeepFixedChannels(montage_channels,keeper_channels)
   return channel_remover

channel_dict = {'two_frontal':two_frontal,
                'two_central':two_central,
                'six_referential':six_referential,
                'all_referential':all_referential,
                'epiminder':['T3','P3','Pz','T4','P4'],
                'uneeg':['T3','F7','T4','F8']
                }

if __name__=='__main__':

   path_model, dataset,keeper_channels = get_args()
   config = get_config(path_model)

   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   transforms = init_standard_transforms(all_referential,config['CHANNELS'],
                                        config['WINDOWSIZE'],0,config['FS'])
   if keeper_channels!=all_referential:
      transforms += [KeepFixedChannels(config['CHANNELS'],channel_dict[keeper_channels])]
   torch.set_float32_matmul_precision('high')
   module = get_datamodule(dataset,transforms=transforms,batch_size = 256)
   preds = generate_predictions(model,trainer,module.test_dataloader())
   labels = module.get_labels('Test')
   event_files = module.get_event_files('Test')
   save_preds(path_model,f'pred_{dataset}_{keeper_channels}.csv',preds,labels,event_files)