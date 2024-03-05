
import sys
sys.path.append('../../')
from local_utils import ResNetInstance, TransformerInstance
from local_utils import WindowCutter, MultiMontage
from local_utils import all_referential, all_bipolar, all_average
import pytorch_lightning as pl
import numpy as np
import torch
import os 
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from local_utils import datamoduleRep, datamoduleClemson, datamoduleHash, datamoduleLocal

sys.path.append('/media/moritz/Expansion/other_models/Spikenet_2')
from sleeplib.montages import CDAC_combine_montage
from sleeplib.transforms import cut_and_jitter,channel_flip,extremes_remover
from sleeplib.Resnet_15.model import FineTuning
from sleeplib.config import Config
from sleeplib.datasets import BonoboDataset

def load_model(architecture,path_weights,n_channels,device):
    if architecture =='ResNet':
        model = ResNetInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device(device))
    elif architecture =='Transformer':
        model = TransformerInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device(device))
    return model

def load_model_code_Jun(path_weights,device,config):
    model = FineTuning.load_from_checkpoint(path_weights,
                    lr=config.LR,
                    head_dropout=config.HEAD_DROPOUT,
                    n_channels=config.N_CHANNELS,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    map_location=torch.device(device)
                    )
    return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def init_transforms_Jun(config):
    transform = transforms.Compose([cut_and_jitter(windowsize=config.WINDOWSIZE,max_offset=0,Fq=config.FQ),
                                    extremes_remover(signal_max = 2000, signal_min = 20)])#,CDAC_signal_flip(p=0)])
    return transform

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def get_dataloader_JUN(dataset,batch_size,transforms,montage):
    if dataset == 'Rep':
        module = datamoduleRep(batch_size,transforms)
        path_files = '/media/moritz/internal_expansion/Data/Bonobo/cluster_center/'
        df = pd.read_csv('/media/moritz/Expansion/other_models/Spikenet_2/lut_labelled_20231218.csv',sep=';')
        df[df.Mode=='Test']
        df = df[df.Mode=='Test']
        pos = df[(df.total_votes_received>=8)&(df.fraction_of_yes>=7/8)]
        neg = df[df.fraction_of_yes==0]
        N = min([len(pos),len(neg)])
        df = pd.concat([pos[:N],neg[:N]])
        event_files = df.event_file
        labels = df.fraction_of_yes

    elif dataset =='Clemson':
        module = datamoduleClemson(batch_size,transforms)
        path_files  = '/media/moritz/Expansion/Data/Spikes_clemson_10s/preprocessed_npy'
        event_files = module.get_event_files('Test')
        labels = module.get_labels('Test')
    test_df = pd.DataFrame({'event_file':event_files,'fraction_of_yes':labels})
    Bonobo_test = BonoboDataset(test_df, 
                            path_files, 
                            transform=transforms, 
                            montage = montage
                            )
    dataloader = DataLoader(Bonobo_test, batch_size=128,shuffle=False,num_workers=os.cpu_count())
    return dataloader, event_files, labels

if __name__ == '__main__':
    dataset = 'Rep' # Rep, Loc, Hash, Clemson are the options
    path_model = '../../Models/ResNet_hardmined/'
    path_weights = os.path.join(path_model,'weights.ckpt')
    architecture = 'ResNet'
    storage_channels = all_referential
    windowjitter = 0
    windowsize = 1
    Fs = 128
    batch_size = 256
    device = 'cpu' # or cuda
    print(f'>>>{path_weights}<<<')
    config = Config()
    model = load_model_code_Jun(path_weights,device,config)
    transform = init_transforms_Jun(config)
    montage = CDAC_combine_montage()
    dataloader,event_files,labels = get_dataloader_JUN(dataset,batch_size,transform,montage)
    trainer = pl.Trainer(max_epochs=300, devices=1)
    preds = generate_predictions(model,trainer,dataloader)
    
    result = pd.DataFrame({'event_file':event_files,'pred':preds,'label':labels})
    result.to_csv(os.path.join(path_model,'pred_'+dataset+'.csv'),index=False)
