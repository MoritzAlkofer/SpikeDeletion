import argparse
from utils import init_standard_transforms, KeepFixedChannels
from utils import get_datamodule
from utils import get_config, load_model_from_checkpoint
from utils import all_referential, two_frontal, two_central, six_referential, all_referential
from utils import generate_predictions, init_trainer
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
      montage_dict = {'frontal':['Fp1','Fp2'],
            'parietal':['P3','P4'],
            'occipital':['O1','O2'],
            'temporal':['T3','T4'],
            'central':['C3','C4'],
            'general':['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']}
   
   if channels == 'point_of_interest':
      montage_dict = {'two_frontal':two_frontal,
                'two_central':two_central,
                'six_referential':six_referential,
                'all_referential':all_referential,
                'epiminder':['T3','P3','Pz','T4','P4'],
                'uneeg':['T3','F7','T4','F8']
                }
   return montage_dict

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model',required=True)
    parser.add_argument('--channels', choices=['localized','point_of_interest'],required=True)
    parser.add_argument('--dataset',choices=['Loc','Rep'],required=True)
    args = parser.parse_args()
    return args.path_model, args.channels, args.dataset


if __name__=='__main__':

   path_model, channels, dataset = get_args()
   
   config = get_config(path_model)
   model = load_model_from_checkpoint(path_model,config)
   trainer = init_trainer()
   torch.set_float32_matmul_precision('high')
   location_dict = init_location_dict(channels)

   transforms = init_standard_transforms(all_referential,config['CHANNELS'],
                                        config['WINDOWSIZE'],0,config['FS'])

   results = {'event_file':[],'fraction_of_yes':[],'pred':[],'ChannelLocation':[]}
   for location,keeper_channels in tqdm(location_dict.items()):
        channel_remover = KeepFixedChannels(config['CHANNELS'],keeper_channels)

        module = get_datamodule(dataset,transforms=transforms+[channel_remover],batch_size=256)
        
        preds = generate_predictions(model,trainer,module.test_dataloader())
        
        results['event_file']+=module.get_event_files('Test')
        results['fraction_of_yes']+=module.get_labels('Test')
        results['pred']+=list(preds)
        results['ChannelLocation']+=[location]*len(preds)

results = pd.DataFrame(results)
results.to_csv(path_model+f'/results_{dataset}_{channels}.csv',index=False)