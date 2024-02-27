import numpy as np
import sys
sys.path.append('../../Scripts')
from local_utils import cut_and_jitter,EDFLoader, StandardFilter
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def center(signal):
    for i in range(signal.shape[0]):
        signal[i,:]=signal[i,:] - np.mean(signal[i,:])
    return signal

print('do clemson')
# edf stuff
path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/data_raw'
files = [ f for f in os.listdir(path_folder) if f.split('.')[-1]=='edf']
path_files_clemson = [os.path.join(path_folder,f) for f in files]

channels = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 'fz', 'cz', 'pz', 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4']
loader = EDFLoader(storage_channels=channels,Fs_up=128)
print(f'its {len(path_files_clemson)} files!')

mean_percentiles_edf = []

for path_file in tqdm(path_files_clemson):
    signal = loader(path_file)
    signal = center(signal)
    percentiles = np.percentile(np.abs(signal), 95, axis=1)
    mean_percentile = np.mean(percentiles)
    mean_percentiles_edf.append(mean_percentile)

print('do Bonobo')
path_folder = '/media/moritz/Expansion/Data/bonobo/raw/cluster_center'
files = [ f for f in os.listdir(path_folder) if f.split('.')[-1]=='npy']
path_files_bonobo = [os.path.join(path_folder,f) for f in files]

mean_percentiles_npy = []
print(f'its {len(path_files_bonobo)} files!')

for path_file in tqdm(path_files_bonobo):
    signal = np.load(path_file)
    signal = center(signal)
    percentiles = np.percentile(np.abs(signal), 95, axis=1)
    mean_percentile = np.mean(percentiles)
    mean_percentiles_npy.append(mean_percentile)

plt.hist(mean_percentiles_npy,bins=np.arange(0,200,5),density=True,label=f'Clemson N={len(path_files_clemson)}')
plt.hist(mean_percentiles_edf,bins=np.arange(0,200,5),density=True,label=f'Bonobo N={len(path_files_bonobo)}')
plt.xlabel('Mean Amplitude of 95th Percentile (μV)')
plt.ylabel('Normalized count')
plt.legend()
plt.savefig('histogram.png')
plt.show(block=True)