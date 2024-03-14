import argparse
from local_utils import init_standard_transforms, KeepFixedChannels
from local_utils import get_datamodule
from local_utils import get_config, load_model_from_checkpoint
from local_utils import points_of_interest, localized_channels, all_referential
from local_utils import generate_predictions, init_trainer
import numpy as np
import os
import pickle
import pandas as pd
#from utils import *
from tqdm import tqdm 
import torch
import pytorch_lightning as pl

def init_location_dict(channels):
   if channels == 'localized':
      # init channel and spike locations
      montage_dict = localized_channels
   
   if channels == 'point_of_interest':
      montage_dict = points_of_interest
   return montage_dict

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model',required=True)
    parser.add_argument('--channels', choices=['localized','point_of_interest'],required=True)
    parser.add_argument('--dataset',choices=['Loc','Rep','Clemson'],required=True)
    parser.add_argument('--split',default='Test',choices=['Val','Test'],required=True)
    args = parser.parse_args()
    return args.path_model, args.channels, args.dataset, args.split


if __name__=='__main__':

   path_model, channels, dataset,split = get_args()
   
   config = get_config(path_model)
   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   location_dict = init_location_dict(channels)

   transforms = init_standard_transforms(all_referential,config['CHANNELS'],
                                        config['WINDOWSIZE'],0,config['FS'])

   results = {'event_file':[],'label':[],'pred':[],'SpikeLocation':[],'ChannelLocation':[]}
   for location,keeper_channels in tqdm(location_dict.items()):
      channel_remover = KeepFixedChannels(config['CHANNELS'],keeper_channels)

      module = get_datamodule(dataset,transforms=transforms+[channel_remover],batch_size=256)
      if split =='Test':
         preds = generate_predictions(model,trainer,module.test_dataloader())
      elif split =='Val':
         preds = generate_predictions(model,trainer,module.val_dataloader())
      
      results['event_file']+=module.get_event_files(split)
      results['label']+=module.get_labels(split)
      results['pred']+=list(preds)
      results['ChannelLocation']+=[location]*len(preds)
      results['SpikeLocation']+=module.get_locations(split)

results = pd.DataFrame(results)
results.to_csv(path_model+f'/results_{dataset}_{channels}_{split}.csv',index=False)