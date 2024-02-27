from BIOViewer import EventViewer
from BIOViewer import EventConfig
import os
import sys
import pandas as pd
sys.path.append('../../Scripts')
from local_utils import cut_and_jitter,EDFLoader, StandardFilter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from make_datamodule import event_dataset
from local_utils import all_referential, build_montage, normalize, cut_and_jitter

class keep_fixed_number_of_fixed_channels():
    # init with total number of channels + channels to be retained
    # use: input, signal, list of channels to be retained
    # output: zero masked signal with *channels from list retained*
    def __init__(self,montage_channels,keeper_channels):
        self.montage_channels = montage_channels
        self.keeper_channels = keeper_channels
      #   print('keeping the following channels: '+keeper_channels)
    def __call__(self,signal):
        keeper_indices = np.array([self.montage_channels.index(channel) for channel in self.keeper_channels])        
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output


def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config


class build_loader():
    def __init__(self,dataset):
        self.dataset = dataset
    def __call__(self,path_file):
        index = self.dataset.filepaths.index(path_file)
        x,y =  dataset[index]
        return x

if __name__=='__main__':

    Fs = 128

    channels = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 'fz', 'cz', 'pz', 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4']
    
    dataset = 'clemson'
    if dataset == 'clemson':
        path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/data_raw'
        path_files = [os.path.join(path_folder,f) for f in os.listdir(path_folder)]
        loader = EDFLoader(storage_channels=channels,Fs_up=Fs)

    if dataset == 'bonobo':
        df = pd.read_csv('../../tables/split_local_13FEB24.csv')
        path_files = df.path.to_list()
        loader = np.load

    path_model = '../../Models/generalized_all_ref_loc'
    config = get_config(path_model)

    montage = build_montage(config.CHANNELS,all_referential)
    cutter = cut_and_jitter(config.WINDOWSIZE,0,config.FQ)
    channel_remover = keep_fixed_number_of_fixed_channels(config.CHANNELS,config.CHANNELS)
    transforms = [montage,cutter,normalize]


    dataset = event_dataset(path_files,[0]*len(path_files),loader)

    loader = build_loader(dataset)

    signalconfig = EventConfig(path_files,loader,Fs,channels,scale= 1.5,unit='mv',transforms=transforms)
    viewer = EventViewer(signal_configs=[signalconfig],windowsize=10,path_save='Figures')