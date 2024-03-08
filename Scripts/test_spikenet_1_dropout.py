import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import os
import numpy as np
import scipy.io as sio
import hdf5storage as hs
from tqdm import tqdm
from mne.filter import notch_filter, filter_data
from keras.models import model_from_json
from make_datamodule import datamoduleLocal, get_split_dfLocal, datamoduleClemson, datamodule, get_split_df
from utils import cut_and_jitter, remove_channels

notch_freq = [60]              
bp_freq    = [0.5, None]      
mono_channels    = ['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ','FP2','F4','C4','P4','F8','T4','T6','O2']
CAR_channels     = ['FP1-CAR','F3-CAR','C3-CAR','P3-CAR','F7-CAR','T3-CAR','T5-CAR','O1-CAR','FZ-CAR','CZ-CAR','PZ-CAR','FP2-CAR','F4-CAR','C4-CAR','P4-CAR','F8-CAR','T4-CAR','T6-CAR','O2-CAR']
bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']

def preprocess_eeg(X, Fs):
    """
    X: (#ch, T) numpy array
    Fs: Sampling frequency in Hz
    Returns: (#ch, T) numpy array
    """
    X = filter_data(X, Fs, bp_freq[0], bp_freq[1], n_jobs=-1, method='fir', verbose=False)
    X = notch_filter(X, Fs, notch_freq, n_jobs=-1, method='fir', verbose=False)
 
    return X

def montage(signal):
    bipolar_ids  = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = signal[bipolar_ids[:,0]]-signal[bipolar_ids[:,1]]
    average_data = signal-signal.mean(axis=0)
    
    signal = np.concatenate([
        average_data,       # Average  - !modify if the initial data is not c2 monopolar #
        bipolar_data,       # L-bipolar #
    ], axis=0)
    return signal

if __name__=='__main__':

    Fs = 128              # sampling rate [Hz]
    step = 8              # step size  
    n_gpu = 0           # 0 or 1 for CPU/GPU, automatically decided based on your computer setup, >1 for multiple GPUs
    batch_size = 1000     # fixed  
    L = int(round(1*Fs))  # the time window this model works, fixed to 1s  

    # load model #
    with open('../Models/Spikenet_1/model_fold 1_structure.txt','r') as fff:
        json_string = fff.read()  
    model = model_from_json(json_string)
    model.load_weights('../Models/Spikenet_1/model_fold 1_weights.h5')
    
    
    module = datamodule(transforms=None,batch_size=batch_size)
    dataloader = module.build_dataloader(mode = 'Test',torch=False)
    df = get_split_df(module.df,'Test')

    cutter = cut_and_jitter(1,0,Fs)

    data,labels = next(iter(dataloader))
    labels = labels.detach().numpy().round(0).astype(int)

    results = {'run':[],'n_keeper':[],'AUROC':[]}
    n_channels = len(mono_channels+bipolar_channels)
    runs = 3
    for n_keeper in tqdm(range(n_channels)):
        for run in tqdm(range(runs),leave=False):
            channel_remover = remove_channels(n_channels,N_keeper=n_keeper)
            preds = []
            for i in tqdm(range(data.shape[0]),leave=False):
                X = data.detach().numpy()
                X = X[i]
                X = cutter(X)
                X = X[:19,:]
                X = montage(X)
                X = channel_remover(X)
                    # preprocess #                    
                X = X.transpose(1,0)
                X = np.expand_dims(X ,axis=1)
                X = np.expand_dims(X ,axis=0)
                pred = model.predict(X)[0][0]
                preds.append(pred)

            fpr, tpr, thresholds = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            results['run'].append(run)
            results['n_keeper'].append(n_keeper)
            results['AUROC'].append(roc_auc)

results = pd.DataFrame(results)
results.to_csv('results_dropout.csv',index=False)