import os
import numpy as np
import scipy.io as sio
import hdf5storage as hs
from tqdm import tqdm
from mne.filter import notch_filter, filter_data
from keras.models import model_from_json
import sys
import pandas as pd
import numpy as np
sys.path.append('/home/moritz/Desktop/programming/SpikeNet_Demo')


notch_freq = [60]              
bp_freq    = [0.5, None]      
mono_channels    = ['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ','FP2','F4','C4','P4','F8','T4','T6','O2']
CAR_channels     = ['FP1-CAR','F3-CAR','C3-CAR','P3-CAR','F7-CAR','T3-CAR','T5-CAR','O1-CAR','FZ-CAR','CZ-CAR','PZ-CAR','FP2-CAR','F4-CAR','C4-CAR','P4-CAR','F8-CAR','T4-CAR','T6-CAR','O2-CAR']
bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']

def read_npy_file(path):
    #TODO You need to custom this function for your data format.
    #TODO Make sure the returned result has the following format.
    """
    path: path to the input mat file.
    Returns: dict, {'Fs': a number, 'data': len(all_channels)xT numpy array}
    """
        
    seg = np.load(path)                 
    seg = seg[0:19,:]    
    seg = np.where(np.isnan(seg), 0, seg)  # deal with NaNs #
                                
    # prepare into both average and L-bipolar #
    bipolar_ids  = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = seg[bipolar_ids[:,0]]-seg[bipolar_ids[:,1]]
    average_data = seg-seg.mean(axis=0)
    
    seg = np.concatenate([
        average_data,       # Average  - !modify if the initial data is not c2 monopolar #
        bipolar_data,       # L-bipolar #
    ], axis=0)
    return seg

def preprocess_eeg(X, Fs):
    """
    X: (#ch, T) numpy array
    Fs: Sampling frequency in Hz
    Returns: (#ch, T) numpy array
    """
    X = filter_data(X, Fs, bp_freq[0], bp_freq[1], n_jobs=-1, method='fir', verbose=False)
    X = notch_filter(X, Fs, notch_freq, n_jobs=-1, method='fir', verbose=False)
 
    return X

if __name__=='__main__':

    root = "/media/moritz/Expansion/Data/Spikes_clemson_10s/"
    dataDir = root+'preprocessed_npy/'
    saveDir = root+'pred_spikenet_v1'

    files = os.listdir(dataDir)

    Fs = 128              # sampling rate [Hz]
    L = int(round(1*Fs))  # the time window this model works, fixed to 1s  

    # load model #
    with open('/home/moritz/Desktop/programming/SpikeNet_Demo/SSD/model_fold 1_structure.txt','r') as fff:
        json_string = fff.read()  
    model = model_from_json(json_string)
    model.load_weights('/home/moritz/Desktop/programming/SpikeNet_Demo/SSD/model_fold 1_weights.h5')

    result = {'event_file':[],'pred':[]}
    for fn in tqdm(files):
        input_path = dataDir + fn

        # load data #
        data = read_npy_file(input_path)
        # preprocess #            
        eeg = preprocess_eeg(data, Fs)
        # cut to shape #
        center = data.shape[1]//2
        data = data[:,center-L//2:center+L//2]

        data = data.transpose(1,0)
        data = np.expand_dims(data,axis=0)
        data = np.expand_dims(data,axis=2)

        # scan with model #
        pred = model.predict(data).flatten().squeeze()
        result['event_file'].append(fn), result['pred'].append(pred)

result = pd.DataFrame(result)
result.pred = result.pred.astype(float)
result.to_csv('preds_spikenet_v1.csv',index=False)
