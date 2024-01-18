import numpy as np
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass, field
import pickle
import os

class build_montage():
    # this version can also convert cdac monopolar montage into mgh_psg monopolar montage
    def __init__(self,montage_channels,storage_channels,echo=True):
        
        # AVERAGE MONTAGE
        # get list of all channels that should be displayed in average montage
        avg_channels = [channel for channel in montage_channels if 'avg' in channel]
        # get ids of channels 

        self.avg_ids = np.array([storage_channels.index(channel.replace('-avg',''), regex=True) for channel in avg_channels])

        # BIPOLAR MONTAGE
        # get list of all channels that should be displayed in average montage
        bipolar_channels = [channel for channel in montage_channels if ('avg' not in channel)&('-' in channel)]
        # get ids of channels 
        self.bipolar_ids = np.array([[storage_channels.index(channel.split('-')[0]), storage_channels.index(channel.split('-')[1])] for channel in bipolar_channels])
    
        # conversion
        # get list of all channels that should be displayed in average montage
        monopolar_channels = [channel for channel in montage_channels if ('avg' not in channel) and ('-' not in channel)]
        # get ids of channels 
        self.monopolar_ids = np.array([storage_channels.index(channel) for channel in monopolar_channels])
    
        if echo: print('storage channels: '+str(storage_channels))
        if echo: print('montage channels: '+str(avg_channels+bipolar_channels+monopolar_channels))

    def __call__(self,signal):
        signals = []
        # AVERAGE MONTAGE
        # get average of these signals along time axis
        if len(self.avg_ids>0):
            avg_signal = signal[self.avg_ids].mean(axis=0).squeeze()
            # substract average from original signal
            avg_montaged_signal = signal[self.avg_ids] - avg_signal
            signals.append(avg_montaged_signal)
        if len(self.bipolar_ids)>0:
            # BIPOLAR MONTAGE
            bipolar_montaged_signal = signal[self.bipolar_ids[:,0]] - signal[self.bipolar_ids[:,1]]
            signals.append(bipolar_montaged_signal)
        if len(self.monopolar_ids>0):
            # add monopolar channels
            signals.append(signal[self.monopolar_ids])

        signal = np.concatenate(signals)
        return signal

class cut_and_jitter():
    def __init__(self,windowsize,max_offset,Fq):
        self.windowsize = int(windowsize*Fq)
        self.max_offset = int(max_offset*Fq)

    def __call__(self,signal):
        # get index of center
        center = signal.shape[1]//2
        # get index of window start
        start = center - (self.windowsize)//2
        # shift by up -1 or 1 x offsetsize
        start = start + int(np.random.uniform(-1, 1)*self.max_offset)
        return signal[:,start:start+self.windowsize]
        
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
        
class SpikeDataset(torch.utils.data.Dataset):
    def __init__(self,df,path_folder, montage=None, windowcutter = None,transform=None,normalize=True,echo=True):
        # set lookup table
        self.df = df
        # set transform
        self.transform = transform
        # set windowcutter
        self.windowcutter = windowcutter
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.normalize = normalize
        if echo:
            if self.normalize: print('Dataloader normalizes signal!\n')
            else: print('Dataloader does not normalize singal!\n')

    def __len__(self):
        return len(self.df)

    def _preprocess(self,signal):
        # convert to desired montage
        signal = self.montage(signal)
        # cut window to desired shape
        signal = self.windowcutter(signal)
        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)                
        if self.normalize==True:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        # get name and label of the idx-th sample
        event_file = self.df.iloc[idx]['event_file']
        label = self.df.iloc[idx]['fraction_of_yes']
        # load signal of the idx-th sample
        path_signal = os.path.join(self.path_folder,event_file+'.npy')
        signal = np.load(path_signal)
        # preprocess signal
        signal = self._preprocess(signal)
        # return signal          
        return signal,label

all_referential = ['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']

full_bipolar = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                             'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                             'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                             'Fz-Cz', 'Cz-Pz']
        
four_bipolar = mgh_psg_bipolar_montage = ['F3-C3','C3-O1','F4-C4','C4-O2']

two_referential = ['Fp1','Fp2']

@dataclass
class Config:

    # Signal parameters
    FQ: int = 128 # Hz
    
    # Preprocessing 
    WINDOWSIZE: int = 10 # seconds
    WINDOWJITTER: int = 2.5 # seconds
    # weird thing to add list to dataclass
    # solution found here: https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    CHANNELS: list = field(default_factory=lambda: all_referential)

	# Model parameters
    N_FFT: int = 128
    HOP_LENGTH: int = 64
    HEAD_DROPOUT: int = 0.3
    EMB_SIZE: int = 256
    HEADS: int = 8
    DEPTH: int = 4
    WEIGHT_DECAY: float = 1e-4
    SIGNAL_MAX: int = 1000
    SIGNAL_MIN: int = 20

    # training parameters
    BATCH_SIZE: int = 32
    LR: float = 1e-4


    def print_config(self):
        print('THIS CONFIG FILE CONTAINS THE FOLLOWING PARAMETERS :\n')
        for key, value in self.__dict__.items():
            print(key, value)
        print('\n')

    def save_config(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        # save config to model_path using pickle
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)     

