import numpy as np
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass, field
import pickle
import pandas as pd
import os

# transforms
class build_montage():
    # this version can also convert cdac monopolar montage into mgh_psg monopolar montage
    def __init__(self,montage_channels,storage_channels,echo=False):
        
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
        
def normalize(signal):
    signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
    signal = signal - np.mean(signal)
    
    return signal

class remove_channels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,N_channels,N_keeper):
        self.N_channels = N_channels
        self.N_keeper = N_keeper
    def __call__(self,signal):
        # choose n random keeper_channels
        if self.N_keeper!='random':
            keeper_indices = np.random.choice(self.N_channels, self.N_keeper, replace=False)
        elif self.N_keeper == 'random':
            N_random_keeper = np.random.randint(1,self.N_channels)
            
            # choose n random channels
            keeper_indices = np.random.choice(self.N_channels, N_random_keeper, replace=False)
        # build output
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output


def get_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fq,random_delete):
    montage = build_montage(montage_channels,storage_channels)
    cutter = cut_and_jitter(windowsize,windowjitter,Fq)
    if random_delete:
        channel_remover = remove_channels(len(montage_channels),N_keeper='random')
        transforms = [montage,cutter,normalize,channel_remover]
        print('deleting random channels!')
    else:
        transforms = [montage,cutter,normalize]
        print('keeping all channels')
    return transforms
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

all_bipolar = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                             'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                             'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                             'Fz-Cz', 'Cz-Pz']
        
six_bipolar = ['F3-C3','C3-O1','F4-C4','C4-O2']

two_referential = ['Fp1','Fp2']

six_referential = ['F3','C3','O1','F4','C4','O2']

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
    BATCH_SIZE: int = 128
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

def dataset_center():
    # returns high confidence dataframe, path and storage channels
    # Load Bonobo dataframe and add 'dataset' column
    #df = pd.read_csv('../Data/tables/archive/lut_event_23-08-22.csv')
    df = pd.read_csv('../Data/tables/lut_event_23-08-22.csv')
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

def dataset_control(N=40000):
    # random controls
    print(N)
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

class ChannelFlipper():

    def __init__(self, p, channels):
        self.p = p
        # get flipped order of channel strings
        if '-' in channels[0]:
            channels_flipped = self._bipolar_flipper(channels)
        else: 
            channels_flipped = self._monopolar_flipper(channels)
        
        # convert strings to indices, this is later applied to the signal
        self.flipped_order = [channels.index(c) for c in channels_flipped]
        # print output to check
        print('used channels:    '+ str(channels))
        print('flipped channels: '+ str(channels_flipped))

    def _flip_channel(self,channel):
        '''
        this function flips a channel string, e.g. Fp1 -> Fp2, F3 -> F4, etc.
        Central channels are kept in place, e.g. Fz -> Fz
        '''
        loc = channel[-1]
        # keep central channels in place
        if loc == 'z': 
            return channel
        # flip all other channels
        else: loc = int(loc)
        if loc % 2 ==1: # +1 all uneven channels. Fp1 -> Fp2, F3 -> F4, etc.
            loc +=1
        else: # -1 all even channels. Fp2 -> Fp1, F4 -> F3, etc.
            loc -= 1
        # recompose channel string
        channel = channel[:-1] + str(loc)
        return channel

    def _monopolar_flipper(self,channels):
        ''' flip all channels in a list of monopolar channels'''
        channels_flipped = []
        for channel in channels:
            channel = self._flip_channel(channel)
            channels_flipped.append(channel)
        return channels_flipped
        
    def _bipolar_flipper(self,channels):
        ''' for both channels in bipolar channel, flip them separately
        keeps zentral in place '''
        channels_flipped = []
        for bipolar_channel in channels:
            channel1, channel2 = bipolar_channel.split('-')
            channel1, channel2 = self._flip_channel(channel1), self._flip_channel(channel2)
            channels_flipped.append(channel1+'-'+channel2) # recompose bipolar channel
        return channels_flipped
    
    def __call__(self, signal):
        # if random number is smaller than p, flip channels 
        if np.random.random() < self.p:
            signal = signal[self.flipped_order,:]
        return signal
    
# this function removes individual channels with extreme values (both too much and too little)
class ExtremesRemover():
    def __init__(self,signal_max = 2000, signal_min = 20):
        self.signal_max = signal_max #microV
        self.signal_min = signal_min #microV

    def __call__(self,signal):
        for channel in range(signal.shape[0]):
            # get peak to peak            
            PP = np.abs(signal[channel,:].max() - signal[channel,:].min())
            # blank out artifact channel wise
            if (PP<self.signal_min) | (PP>self.signal_max):
                signal[channel,:] = 0
        return signal
    
class RandomScaler():
    def __init__(self,scale_percent):
        self.scale_percent = scale_percent

    def __call__(self,signal):
        scale_factor = 1+np.random.uniform(-self.scale_percent, self.scale_percent)
        signal = scale_factor*signal
        return signal
        