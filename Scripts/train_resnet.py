import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance
from local_utils import Montage, WindowCutter, KeepRandomChannels
from local_utils import all_referential
from local_utils import datamoduleRep, datamoduleClemson, datamoduleHash, datamoduleLocal
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def init_transforms(random_drop,montage_channels,storage_channels,windowsize,windowjitter,Fs):
    montage = Montage(montage_channels,storage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    if random_drop == False:
        return [montage,cutter]
    else:
        dropper = KeepRandomChannels(len(montage_channels))
        return [montage,cutter,dropper]

def init_callbacks(path_model):
    callbacks = [EarlyStopping(monitor='val_loss',patience=20),ModelCheckpoint(dirpath=path_model,filename='weights',monitor='val_loss',mode='min')]
    return callbacks

def get_model(architecture,n_channels,lr,weight_decay):
    if architecture == 'ResNet':
        model = ResNetInstance(n_channels=n_channels,lr=lr,weight_decay=weight_decay)
    if architecture == 'Transformer':
        model = TransformerInstance(lr=lr,
                                    weight_decay=weight_decay,
                                    n_channels=n_channels)
    print('>>> hardcoded model parameters in get_model <<<')
    return model

def get_datamodule(dataset,transforms,batch_size):
    if dataset =='Rep':
        module = datamoduleRep(transforms=transforms,batch_size=batch_size)
    elif dataset == 'Loc':
        module = datamoduleLocal(transforms,batch_size)
    elif dataset == 'Clemson':
        module = datamoduleClemson(transforms,batch_size)
    elif dataset == 'Hash':
        module = datamoduleHash(transforms,batch_size)
    else: 
        raise ValueError('Please specify dataset correctly! Options are: Rep, Loc, Clemson, Hash')
    return module

if __name__ == '__main__':
    storage_channels = all_referential
    montage_channels = all_referential
    windowsize = 10
    windowjitter = 2.5
    Fs = 128
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-4
    dataset = 'Rep' # Rep, Loc, Hash, Clemson are the options
    path_model = '../Models/ResNet'
    random_drop=False

    transforms = init_transforms(random_drop,montage_channels,storage_channels,windowsize,windowjitter,Fs)
    modelinstance = get_model('ResNet',len(montage_channels),lr,weight_decay)
    module = get_datamodule(dataset,transforms,batch_size)
    callbacks = init_callbacks(path_model=path_model)
    wandb_logger = WandbLogger(project='SpikeTransformer') 
    trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, callbacks=callbacks, devices=1)
    trainer.fit(modelinstance,datamodule=module)

