from BIOViewer import EventViewer
from BIOViewer import EventConfig
import os
import sys
import pandas as pd
sys.path.append('../../Scripts')
from local_utils import cut_and_jitter,EDFLoader, StandardFilter
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':

    channels = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 'fz', 'cz', 'pz', 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4']
    Fs = 128
    path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/npy_preprocessed'
    df = pd.read_csv('../../Models/generalized_all_ref_loc/results_localized_clemson.csv')
    df = df[df.ChannelLocation=='frontal']
    titles = df.pred.to_list()
    path_files = [os.path.join(path_folder,f) for f in df.event_file]
    filter = StandardFilter(Fs,notch_freq=60,band_low=0.5,band_high=60)
    loader = np.load

    cutter = cut_and_jitter(windowsize=10,max_offset=0,Fq=128)

    signalconfig = EventConfig(path_files,loader,Fs,channels,scale= 20,unit='mv',transforms=[cutter])
    viewer = EventViewer(signal_configs=[signalconfig],windowsize=10,path_save='Figures',title=titles)