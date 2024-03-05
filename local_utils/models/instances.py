from pytorch_lightning import LightningModule
from torch.nn import BCELoss
import torch.nn as nn
import torch
from torch.nn import BCELoss
from .transformer import BIOTEncoder
from .heads import RegressionHead
from .Resnet_15.net1d import Net1D
from .Spikenet import SpikeNet

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,target,pred):
        return torch.sqrt(self.mse(target,pred))

# create an instance base class
class InstanceBaseClass(LightningModule):
    def __init__(self,model,head,lr,weight_decay):
        super().__init__()
        self.lr = lr
        self.weight_decay=weight_decay
        # create the BIOT encoder
        self.model = model
        # create the regression head
        self.head = head
        self.RMSE = RMSELoss()
        self.loss = BCELoss()

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

    def training_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('train_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('train_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('val_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        preds = self.forward(signals)
        # compute and log loss
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        return optimizer
    
class TransformerInstance(InstanceBaseClass):
    def __init__(self,n_channels,lr=1e-4,head_dropout=0.3,weight_decay=1e-4,emb_size=256,heads=8,depth=4,n_fft=128,hop_length=64):
        model = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth,n_channels=n_channels,n_fft=n_fft,hop_length = hop_length)
        head = RegressionHead(emb_size,head_dropout)
        super().__init__(model,head,lr,weight_decay)

class ResNetInstance(InstanceBaseClass):
    def __init__(self,n_channels,lr=1e-4,weight_decay=1e-4):
        model = Net1D(
                    in_channels=n_channels, 
                    base_filters=64, 
                    #base_filters=32, 
                    ratio=1, 
                    #filter_list = [64,128,128,160,160,256,256],#
                    filter_list=[64,160,160,400,400,1024,1024], 
                    m_blocks_list=[2,2,2,3,3,4,4], 
                    kernel_size=16, 
                    stride=2, 
                    groups_width=16,
                    verbose=False, 
                    use_bn=True,
                    return_features=False, #False
                    #n_classes=1, #soft target
                    n_classes=1 #hard target
        )
        print('this model comes with A TON of random hardcoded stuff!')
        head = self._identity
        super().__init__(model,head,lr,weight_decay)
        
    #this implementation comes with an integrated head
    def _identity(self,x):
        return x
    
class SpikeNetInstance(InstanceBaseClass):
    def __init__(self,n_channels,lr=1e-4,weight_decay=1e-4):
        model = SpikeNet(eeg_channels=n_channels)
        print('this model comes with A TON of random hardcoded stuff!')
        head = self._identity
        super().__init__(model,head,lr,weight_decay)
        
    #this implementation comes with an integrated head
    def _identity(self,x):
        return x