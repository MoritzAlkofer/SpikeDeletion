import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance
from local_utils import WindowCutter, MultiMontage
from local_utils import all_referential, all_bipolar, all_average
import pytorch_lightning as pl
import numpy as np
import torch
import os 
import pandas as pd
from local_utils import datamoduleRep, datamoduleClemson, datamoduleHash, datamoduleLocal

def load_model(architecture,path_weights,n_channels):
    if architecture =='ResNet':
        model = ResNetInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    elif architecture =='Transformer':
        model = TransformerInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs):
    montage = MultiMontage(montage_channels,storage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    return [montage,cutter]

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds


def get_datamodule(dataset,transforms,batch_size):
    if dataset =='Rep':
        module = datamoduleRep(transforms=transforms,batch_size=batch_size)
    elif dataset == 'Loc':
        module = datamoduleLocal(transforms,batch_size)
    elif dataset == 'Clemson':
        module = datamoduleClemson(transforms,batch_size)
    elif dataset == 'Hash':
        module = datamoduleHash(transforms,batch_size)
    else: 
        raise ValueError('Please specify dataset correctly! Options are: Rep, Loc, Clemson, Hash')
    return module

if __name__ == '__main__':
    dataset = 'Rep' # Rep, Loc, Hash, Clemson are the options
    path_model = '../Models/ResNet/'
    path_weights = os.path.join(path_model,'weights.ckpt')
    architecture = 'ResNet'
    storage_channels = all_referential
    montage_channels = all_bipolar
    windowsize = 10
    windowjitter = 0
    Fs = 128
    n_channels = len(montage_channels)
    print(f'>>>{path_weights}<<<')
    model = load_model(architecture,path_weights,n_channels)
    transforms = init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs)
    
    print('there is an increase added to the trasforms!')
    module = get_datamodule(dataset,transforms,batch_size=128)
    dataloader = module.test_dataloader()
    trainer = pl.Trainer(max_epochs=300, devices=1)
    preds = generate_predictions(model,trainer,dataloader)
    
    event_files, labels = module.get_event_files('Test'), module.get_labels('Test')
    result = pd.DataFrame({'event_file':event_files,'pred':preds,'label':labels})
    result.to_csv(os.path.join(path_model,'pred_'+dataset+'.csv'),index=False)
