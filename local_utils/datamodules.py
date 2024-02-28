import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import os
import pandas as pd
import sys

# event dataset
class event_dataset():
    def __init__(self, path_files, Y, load_signal, signal_transforms=None):
        self.path_files = path_files
        self.event_files = [f.split('/')[-1] for f in self.path_files]
        self.Y = Y
        self.load_signal = load_signal
        self.signal_transforms = [] if signal_transforms == None else signal_transforms

    def __len__(self):
        return len(self.path_files)
    
    def __getitem__(self,idx):
        filepath = self.path_files[idx]
        y = self.Y[idx]
        x = self.load_signal(filepath)
        for transform in self.signal_transforms:
            x = transform(x)
        return x, y   

class DatamoduleBase(pl.LightningDataModule):
    def __init__(self,path_files,labels,modes,loader,transforms,batch_size,collate_fn=None):
        self.path_files = path_files
        self.event_files = [f.split('/')[-1] for f in path_files]
        self.labels = labels
        self.loader = loader
        self.modes = modes
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def build_dataloader(self,mode):
        path_files = self.get_path_files(mode)
        labels = self.get_labels(mode)
        shuffle = True if mode == 'Train' else False
        dataset = event_dataset(path_files,labels, self.loader, signal_transforms=self.transforms)
        dataloader = DataLoader(dataset,self.batch_size,shuffle=shuffle,num_workers=os.cpu_count(),persistent_workers=True, collate_fn=self.collate_fn)
        return dataloader

    def train_dataloader(self):        
        dataloader = self.build_dataloader('Train')
        return dataloader
    
    def val_dataloader(self):
        dataloader = self.build_dataloader('Val')
        return dataloader
    
    def test_dataloader(self):
        dataloader = self.build_dataloader('Test')
        return dataloader
    
    def get_labels(self,mode):
        msk = np.array(self.modes) == mode
        labels = np.array(self.labels)[msk]
        return labels
    
    def get_path_files(self,mode):
        msk = np.array(self.modes) == mode
        path_files = np.array(self.path_files)[msk]
        return path_files

    def get_event_files(self,mode):
        msk = np.array(self.modes) == mode
        event_files = np.array(self.event_files)[msk]
        return event_files
    
class datamoduleRep(DatamoduleBase):
    def __init__(self,transforms,batch_size):
        df = pd.read_csv('../tables/split_representative_12FEB24.csv')
        loader = np.load
        path_files, labels, modes = [],[],[]
        for mode in ['Train','Val','Test']:
            df_mode = self.get_split_df(df,mode)
            path_files+=df_mode.path.to_list()
            labels+=df_mode.fraction_of_yes.to_list()
            modes+=[mode]*len(df_mode)

        super().__init__(path_files,labels,modes,loader,transforms,batch_size,collate_fn=collate_fn)
    
    def get_split_df(self,df,Mode,echo=False):
        if Mode == 'Train':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if Mode == 'Val':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if (Mode == 'Test'):
            df = df[(df.Mode==Mode)&(df.dataset=='center')]
            pos = df[(df.total_votes_received>=8)&(df.fraction_of_yes>=7/8)]
            neg = df[df.fraction_of_yes==0]
            N = min([len(pos),len(neg)])
            df = pd.concat([pos[:N],neg[:N]])
        df.reset_index(inplace=True)
        if echo:
            ratio = len(df[df.fraction_of_yes>0.5])/len(df)
            print(f'\n{Mode} with {len(df)} samples and label ratio pos/all: {np.round(ratio,2)}')
        return df

class datamoduleLocal(DatamoduleBase):
    def __init__(self,transforms,batch_size,echo=False):
        super().__init__()
        df = pd.read_csv('../tables/split_local_13FEB24.csv')
        path_files, labels, modes = [],[],[]
        for mode in ['Train','Val','Test']:
            df_mode = self.get_split_df(df,mode)
            path_files+=df_mode.path.to_list()
            labels+=df_mode.fraction_of_yes.to_list()
            modes+=[mode]*len(df_mode)
        loader = np.load

        super().__init__(path_files,labels,modes,loader,transforms,batch_size,collate_fn)

    def get_split_dfLocal(df,Mode,echo=False):
        if Mode == 'Train':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if Mode == 'Val':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if (Mode == 'Test'):
            df = df[(df.Mode==Mode)&(df.dataset=='center')]
            pos = df[~df.location.isna()]
            neg = df[df.fraction_of_yes==0]
            df = pd.concat([pos,neg])
        df.reset_index(inplace=True)
        if echo:
            ratio = len(df[df.fraction_of_yes>0.5])/len(df)
            print(f'\n{Mode} with {len(df)} samples and label ratio pos/all: {np.round(ratio,2)}')
        return df

class datamoduleHash(DatamoduleBase):
    def __init__(self,transforms,batch_size):
        path_folder = '/media/moritz/Expansion/Data/bids_spikes/hashfolder/'
        df = pd.read_csv(path_folder+'/spike_location_Brandon_confirmed_23FEB24.csv')
        path_files = [os.path.join(path_folder,'standard_processed_data',event_file+'.npy') for event_file in self.df.event_file]
        labels = [-1]*len(self.paths)
        modes = ['Test']*len(labels)
        loader = np.load
        super().__init__(path_files,labels,modes,loader,transforms,batch_size,collate_fn=collate_fn)

class datamoduleClemson(DatamoduleBase):
    def __init__(self,batch_size,transforms=None,echo=False):
        super().__init__()
        self.df = pd.read_csv('/media/moritz/Expansion/Data/Spikes_clemson_10s/segments_list_complete.csv')
        path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/preprocessed_npy'
        path_files = [os.path.join(path_folder,event_file+'.npy') for event_file in self.df.event_file]
        labels = [-1]*len(path_files)
        modes = 'Test'
        loader = np.load
        super().__init__(path_files,labels,modes,loader,transforms,batch_size,collate_fn=collate_fn)
        
def collate_fn(batch):
    # process your batch
    X, y = zip(*batch)
    X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y