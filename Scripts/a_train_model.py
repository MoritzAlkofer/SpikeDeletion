import argparse
import os
import pandas as pd
import numpy as np
import wandb
import pytorch_lightning as pl
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import EEGTransformer
from utils import *

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint



def dataset_center():
    # returns high confidence dataframe, path and storage channels
    # Load Bonobo dataframe and add 'dataset' column
    df = pd.read_csv('../Data/tables/center_17JAN24.csv')
    df['dataset'] = 'center' 
    df = df[df.total_votes_received>2]
    path = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_center/' 
    storage_channels = all_referential

    return df, path, storage_channels

def dataset_member():
    # cluster member
    df = pd.read_csv('../Data/tables/member_17JAN24.csv')
    df = df[df.total_votes_received>2]
    df['dataset'] = 'member' 
    p = [0.9, 0.1] 
    vals = np.random.choice(['Train', 'Val'], size=len(df), p=p)
    df['Mode']=vals
    df['total_votes_received']=8
    df['event_file'] = df.eeg_file.str.replace('.mat','', regex=True)
    path = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_members/' 
    storage_channels= all_referential

    return df, path, storage_channels

def dataset_control(N=30000):
    # random controls
    df = pd.read_csv('/home/moritz/Desktop/programming/epilepsy_project/tables/bonobo/lut_event_random_controls_23-11-06.csv')
    df['dataset'] = 'control' 
    p = [0.9, 0.1] 
    vals = np.random.choice(['Train', 'Val'], size=len(df), p=p)
    df['Mode']=vals
    if N !="all":
        df = df[:N]

    # Add metadata for Bonobo dataset
    path = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/random_snippets/' 
    storage_channels = all_referential
    return df,path,storage_channels

def prepare_member_and_center_info(datasets):
    # input: list of dataset functions, each of which returns the datasets dataframe, location and storage chanel
    # output: concatenated_dataframes + metadata dictionary
    # Initialize empty list to collect dataframes
    dfs = []
    # Create empty dictionary to store dataset metadata
    metadata = {}

    for dataset in datasets.keys():
        datasets[dataset]()
        df, path, storage_channels = datasets[dataset]()
        dfs.append(df)
        metadata[dataset]={}
        metadata[dataset]['path']=path
        metadata[dataset]['storage_channels']=storage_channels

    df = pd.concat(dfs)
    df = df[['event_file','patient_id','total_votes_received','fraction_of_yes','Mode','dataset']]

    print(f'\n using the following datasets: {metadata.keys()} to build datamodule\n')
    # print how much data each dataset has
    print(df.dataset.value_counts())
    return df, metadata

class DeletedChannelsDatamoduleWithMembers(pl.LightningDataModule):
    def __init__(self,df,metadata,storage_channels,montage_channels,windowsize,windowjitter,Fq,batch_size,n_keeper_channels='random',keeper_channels='random',echo=True):
        super().__init__()
        self.df = df
        self.batch_size = batch_size

        if echo:
            print('building datamodule!')
            print(f'there are {len(df[df.Mode=="Train"])} test samples and {len(df[df.Mode=="Val"])} val samples')
            print(f'the fraction of positive samples is {df.fraction_of_yes.sum()/len(df):.2f}\n')

        channel_deleter = choose_channel_deleter(n_keeper_channels,montage_channels,keeper_channels)        
        # build deleter into dataset as a transformation!
        transforms_all = transforms.Compose([channel_deleter])

        # build montage and windowcutter
        windowcutter = cut_and_jitter(windowsize=windowsize,max_offset=windowjitter,Fq=Fq)

        self.dataset_train = MultiSourceSpikeDataset(df=self.df[self.df.Mode=='Train'],metadata = metadata,montage_channels=montage_channels, windowcutter = windowcutter,transform=transforms_all,normalize=True,echo=False)
        self.dataset_val = MultiSourceSpikeDataset(df=self.df[self.df.Mode=='Val'],metadata = metadata,montage_channels=montage_channels, windowcutter = windowcutter,transform=transforms_all,normalize=True,echo=False)    
        self.dataset_test = MultiSourceSpikeDataset(df=self.df[self.df.Mode=='Test'],metadata = metadata,montage_channels=montage_channels, windowcutter = windowcutter,transform=transforms_all,normalize=True,echo=False)
       
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

class MultiSourceSpikeDataset(Dataset):
    def __init__(self, df, metadata, montage_channels, windowcutter = None,transform=None,normalize=False,echo=True):
        
        if echo: print(f'building dataset from {metadata.keys()}')
        # set lookup table
        self.df = df
        # set metadata
        self.metadata = metadata
        # set transform
        self.transform = transform
        # set windowcutter
        self.windowcutter = windowcutter
        # set normalize
        self.normalize = normalize
        if echo:
            if self.normalize: print('Dataloader normalizes!\n')
            else: print('Dataloader does not normalize!\n')

        # generate montages for all datasets
        self.montages = {}
        for dataset in metadata.keys():
            if echo: print('build montage for dataset ' + dataset + ' ...')
            montage = build_montage(storage_channels=metadata[dataset]['storage_channels'], 
                                                      montage_channels= montage_channels,
                                                      echo=echo)
            self.montages[dataset] = montage
    def __len__(self):
        return len(self.df)

    def _dataset_specific_loading_and_montage(self,event_file,dataset):
        path_signal = os.path.join(self.metadata[dataset]['path'],event_file+'.npy')
        signal = np.load(path_signal)
        signal = self.montages[dataset](signal)    
        return signal
    
    def _preprocess(self,signal):
        # cut window to desired shape
        if self.windowcutter is not None:
            signal = self.windowcutter(signal)
        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)                
        # normalize signal
        if self.normalize==True:
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        # convert to torch tensor
        # replace nan values with 0
        signal = np.nan_to_num(signal)
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        # get name and label of the idx-th sample
        event_file = self.df.iloc[idx]['event_file']
        label = self.df.iloc[idx]['fraction_of_yes']
        # get dataset specific information
        dataset = self.df.iloc[idx]['dataset']

        # load signal of the idx-th sample
        signal =self._dataset_specific_loading_and_montage(event_file,dataset)
        # preprocess signal
        signal = self._preprocess(signal)
        # return signal    
        return signal,label
        
class keep_fixed_number_of_random_channels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,montage_channels,n_keeper_channels):
        self.n_channels = len(montage_channels)
        self.n_keeper_channels = n_keeper_channels
    def __call__(self,signal):
        # choose n random keeper_channels
        keeper_indices = np.random.choice(self.n_channels, self.n_keeper_channels, replace=False)
        # build output
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output

class keep_fixed_number_of_fixed_channels():
    # init with total number of channels + channels to be retained
    # use: input, signal, list of channels to be retained
    # output: zero masked signal with *channels from list retained*
    def __init__(self,montage_channels,keeper_channels):
        self.montage_channels = montage_channels
        self.keeper_channels = keeper_channels
    def __call__(self,signal):
        keeper_indices = np.array([self.montage_channels.index(channel) for channel in self.keeper_channels])        
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output

class keep_random_number_of_random_channels():
    # init with list of montaged channels
    # use: input: signal
    # output: zero masked signal with *random* number of random channels retained
    def __init__(self,montage_channels):
        self.n_channels = len(montage_channels)
    def __call__(self,signal):
        # keep n channels between one and |all channels|
        n_keeper_channels = np.random.randint(1,self.n_channels)
        # choose n random channels
        keeper_indices = np.random.choice(self.n_channels, n_keeper_channels, replace=False)
        # build output
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
        
def choose_channel_deleter(n_keeper_channels,montage_channels,keeper_channels):
    # choose right channel deleter dependent on what type is desired
    if n_keeper_channels=='random':
        print('keeping random number of random channels!\n')
        channel_deleter = keep_random_number_of_random_channels(montage_channels=montage_channels)
    elif (n_keeper_channels!='random')&(keeper_channels=='random'):
        print(f'keeping {n_keeper_channels} random channels\n')
        channel_deleter = keep_fixed_number_of_random_channels(montage_channels=montage_channels,n_keeper_channels=n_keeper_channels)
    elif (n_keeper_channels!='random')&(keeper_channels!='random'):
        print(f'keeping the following channels: {keeper_channels}!\n')
        channel_deleter = keep_fixed_number_of_fixed_channels(montage_channels=montage_channels,keeper_channels=keeper_channels)

    return channel_deleter

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--model_name',default='two_referential')
    parser.add_argument('--channels',default='two_referential', choices=['all_referential','two_referential','all_bipolar','four_bipolar'])
    args = parser.parse_args()
    return args.model_name, args.channels

def str_to_channel_list(channels):
    if channels == 'all_referential':
        return all_referential
    if channels == 'two_referential':
        return two_referential
    if channels == 'all_bipolar':
        return all_bipolar
    if channels == 'four_bipolar':
        return four_bipolar

def init_model(model):
    model = EEGTransformer(lr=config.LR,
                    head_dropout=config.HEAD_DROPOUT,
                    n_channels=len(config.CHANNELS),
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    depth= config.DEPTH,
                    heads=config.HEADS,
                    emb_size=config.EMB_SIZE,
                    weight_decay=config.WEIGHT_DECAY,
                    )
    return model

def init_datamodule_info():
    datasets = {'center':dataset_center,'member':dataset_member,'control':dataset_control}
    df,metadata = prepare_member_and_center_info(datasets)
    storage_channels = all_referential
    return df, metadata, storage_channels

def build_datamodule(df, metadata, storage_channels):
    datamodule = DeletedChannelsDatamoduleWithMembers(df=df,
                                                  storage_channels=storage_channels,
                                                  montage_channels=config.CHANNELS,
                                                  Fq=config.FQ,
                                                  metadata=metadata,
                                                  windowsize=config.WINDOWSIZE,
                                                  windowjitter=config.WINDOWJITTER,
                                                  batch_size=config.BATCH_SIZE,
                                                  n_keeper_channels='fixed',
                                                  keeper_channels=config.CHANNELS)
    return datamodule

def init_callbacks(path_model, patience=20):
    callbacks = [EarlyStopping(monitor='val_loss',patience=patience),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

if __name__ == '__main__':

    model_name, channels = get_args()
    path_model = f'../Models/{model_name}'
    config = Config()
    config.CHANNELS = str_to_channel_list(channels)
    config.save_config(path_model)

    df, metadata, storage_channels = init_datamodule_info()
    datamodule = build_datamodule(df, metadata, storage_channels)
    model = init_model(config)

    # create a logger
    wandb.init(name =model_name, dir='.logging')
    wandb_logger = WandbLogger(project='SpikeTransformer') 

    torch.set_float32_matmul_precision('high')
    callbacks = init_callbacks(path_model)
    # create trainer, use fast dev run to test the code
    trainer = pl.Trainer(max_epochs=300,
                        default_root_dir='./logging',
                        logger=wandb_logger,
                        callbacks=callbacks,
                        devices=1,
                        fast_dev_run=False)
    # train the model
    trainer.fit(model,datamodule)
    wandb.finish()
