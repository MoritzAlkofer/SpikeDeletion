import pandas as pd 
from .channel_lists import all_referential
import pickle
import os
from dataclasses import dataclass, field

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


