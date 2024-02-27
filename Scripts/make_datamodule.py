import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import os
import pandas as pd
import sys
sys.path.append('.')

# load the signal
def load_npy(filepath):
    signal = np.load(filepath)
    # replace nan values with 0
    signal = np.nan_to_num(signal)
    return signal 

# event dataset
class event_dataset():
    def __init__(self, filepaths, Y, load_signal, signal_transforms=None):
        self.filepaths = filepaths
        self.Y = Y
        self.load_signal = load_signal
        self.signal_transforms = [] if signal_transforms == None else signal_transforms

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self,idx):
        filepath = self.filepaths[idx]
        y = self.Y[idx]
        x = self.load_signal(filepath)
        for transform in self.signal_transforms:
            x = transform(x)
        return x, y   

# final datamodule
class datamoduleRep(pl.LightningDataModule):
    def __init__(self,transforms,batch_size,echo=False):
        super().__init__()
        self.df = pd.read_csv('../tables/split_representative_12FEB24.csv')
        self.transforms = transforms
        self.batch_size = batch_size
        self.echo = echo

    def build_dataloader(self,mode):
        df = self.df
        df = get_split_df(df, mode,self.echo)
        paths, fraction_of_yes = df.path.tolist(),df.fraction_of_yes.tolist()
        shuffle = True if mode == 'Train' else False
        dataset = event_dataset(paths,fraction_of_yes, load_npy, signal_transforms=self.transforms)
        dataloader = DataLoader(dataset,self.batch_size,shuffle=shuffle,num_workers=os.cpu_count(),persistent_workers=True, collate_fn=collate_fn)
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
    
def collate_fn(batch):
    # process your batch
    X, y = zip(*batch)
    X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

# apply filters to select correct dataset
def get_split_df(df,Mode,echo=False):
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

class datamoduleHash():
    def __init__(self,transforms,batch_size,echo=False):
        path_folder = '/media/moritz/Expansion/Data/bids_spikes/hashfolder/'
        self.df = pd.read_csv(path_folder+'/spike_location_Brandon_confirmed_23FEB24.csv')
        self.paths = [os.path.join(path_folder,'standard_processed_data',event_file+'.npy') for event_file in self.df.event_file]
        self.fraction_of_yes = [-1]*len(self.paths)
        self.transforms = transforms
        self.batch_size = batch_size
        self.echo = echo
        self.loader = np.load

    def build_dataloader(self,mode):
        if mode !='Test':
            raise Exception('Clemson must be used for testing only!')
        shuffle = False
        dataset = event_dataset(self.paths,self.fraction_of_yes, self.loader, signal_transforms=self.transforms)
        dataloader = DataLoader(dataset,self.batch_size,shuffle=shuffle,num_workers=os.cpu_count(),persistent_workers=True, collate_fn=self.collate_fn)
        return dataloader
            
    def test_dataloader(self):
        dataloader = self.build_dataloader('Test')
        return dataloader

    def collate_fn(self,batch):
        # process your batch
        X, y = zip(*batch)
        X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return X, y



# final datamodule
class datamoduleLocal(pl.LightningDataModule):
    def __init__(self,transforms,batch_size,echo=False):
        super().__init__()
        self.df = pd.read_csv('../tables/split_local_13FEB24.csv')
        self.transforms = transforms
        self.batch_size = batch_size
        self.echo = echo

    def build_dataloader(self,mode):
        df = self.df
        df = get_split_dfLocal(df, mode,self.echo)
        paths, fraction_of_yes = df.path.tolist(),df.fraction_of_yes.tolist()
        shuffle = True if mode == 'Train' else False
        dataset = event_dataset(paths,fraction_of_yes, load_npy, signal_transforms=self.transforms)
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
    
    def collate_fn(self,batch):
        # process your batch
        X, y = zip(*batch)
        X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return X, y
    
class datamoduleClemson(pl.LightningDataModule):
    def __init__(self,batch_size,transforms=None,echo=False):
        super().__init__()
        self.df = pd.read_csv('/media/moritz/Expansion/Data/Spikes_clemson_10s/segments_list_complete.csv')
        path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/preprocessed_npy'
        self.paths = [os.path.join(path_folder,event_file+'.npy') for event_file in self.df.event_file]
        self.fraction_of_yes = [-1]*len(self.paths)
        self.transforms = transforms
        self.batch_size = batch_size
        self.echo = echo
        self.loader = np.load
        
    def build_dataloader(self,mode):
        if mode !='Test':
            raise Exception('Clemson must be used for testing only!')
        shuffle = False
        dataset = event_dataset(self.paths,self.fraction_of_yes, self.loader, signal_transforms=self.transforms)
        dataloader = DataLoader(dataset,self.batch_size,shuffle=shuffle,num_workers=os.cpu_count(),persistent_workers=True, collate_fn=self.collate_fn)
        return dataloader
        
    def test_dataloader(self):
        dataloader = self.build_dataloader('Test')
        return dataloader
    
    def collate_fn(self,batch):
        # process your batch
        X, y = zip(*batch)
        X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return X, y

# apply filters to select correct dataset
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