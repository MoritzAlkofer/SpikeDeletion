import os
import sys
from tqdm import tqdm
import numpy as np
sys.path.append('../../Scripts')
from local_utils import EDFLoader, StandardFilter, all_referential

path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s'
storage_channels = [c.lower() for c in all_referential]
Fs_target = 128


filter = StandardFilter(Fs=Fs_target)
loader = EDFLoader(storage_channels=storage_channels,Fs_up=Fs_target)

files = [f.replace('.edf','') for f in os.listdir((os.path.join(path_folder,'data_raw')))]

for file in tqdm(files):
    path_file = os.path.join(path_folder,'data_raw',file+'.edf')
    data = loader(path_file)
    data = filter(data)
    path_save = os.path.join(path_folder,'preprocessed_npy',file+'.npy')
    np.save(path_save,data)
