import os 
import scipy 
import pandas as pd
import sys
sys.path.append('../Scripts')
from local_utils import StandardFilter
from scipy.signal import resample_poly
from tqdm import tqdm

path_root = '/media/moritz/Expansion/Data/bids_spikes/hashfolder'
df = pd.read_csv(os.path.join(path_root,'spike_location_Brandon_confirmed_23FEB24.csv'))

files=os.listdir(os.path.join(path_root,'raw_data'))

filter = StandardFilter(Fs=128)
df[(df.event_file+'.npy').isin(files)]
import numpy as np
for file in tqdm(files):
    signal = np.load(os.path.join(path_root,'raw_data',file))
    signal = filter(signal)
    signal = resample_poly(signal,up=128,down=200,axis=1)
    path_save = os.path.join(path_root,'standard_processed_data',file)
    np.save(path_save,signal)