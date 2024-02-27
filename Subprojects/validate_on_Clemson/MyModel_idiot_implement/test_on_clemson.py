import os
from tqdm import tqdm
import pickle
import sys 
import numpy as np
sys.path.append('/home/moritz/Desktop/programming/SpikeDeletion/Scripts/')
from local_utils import remove_channels, build_montage, normalize, cut_and_jitter, Config, all_referential, all_bipolar
from model import EEGTransformer
import torch

def get_config(path_model):
   with open(os.path.join(path_model,'config.pkl'), 'rb') as f:
      config = pickle.load(f)
   return config

def load_model_from_checkpoint(path_model,config):
   model = EEGTransformer.load_from_checkpoint(os.path.join(path_model,'weights.ckpt'),
                                          lr=config.LR,
                                          head_dropout=config.HEAD_DROPOUT,
                                          n_channels=len(config.CHANNELS),
                                          n_fft=config.N_FFT,
                                          hop_length=config.HOP_LENGTH,
                                          heads = config.HEADS,
                                          depth=config.DEPTH,
                                          emb_size = config.EMB_SIZE,
                                          weight_decay = config.WEIGHT_DECAY)
   return model

class PreprocessEEG():
    def __init__(self,montage_channels,storage_channels,windowsize,windowjitter,Fs):
        montage = build_montage(montage_channels=montage_channels,storage_channels=storage_channels)
        cutter = cut_and_jitter(windowsize,0,Fs)
        self.transforms = [montage,cutter,normalize]

    def __call__(self,signal):
        for transform in self.transforms:
            signal = transform(signal)
        return signal

if __name__ == '__main__':
    path_model = '/home/moritz/Desktop/programming/SpikeDeletion/Models/specialized_all_bipolar'
    root = "/media/moritz/Expansion/Data/Spikes_clemson_10s/"
    dataDir = root+'preprocessed_npy/'

    

    config = get_config(path_model)
    model = load_model_from_checkpoint(path_model,config)
    model = model.to('cpu')
    preprocess = PreprocessEEG(all_bipolar,all_referential,10,0,128)

    result = {'event_file':[],'pred':[]}
    for file in tqdm(os.listdir(dataDir)):
        signal = np.load(os.path.join(dataDir,file))
        signal = preprocess(signal)
        signal = np.expand_dims(signal,0)
        signal = torch.from_numpy(signal).to(torch.float)
        pred = model(signal).item()
        result['event_file'].append(file),result['pred'].append(pred)

result = pd.DataFrame(result)
result.event_file = result.event_file.str.replace('.npy','')
result.to_csv('result_easy_implement.csv',index=False)