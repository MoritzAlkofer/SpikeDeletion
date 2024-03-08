from torch.nn import BCELoss
from pytorch_lightning import LightningModule
import torch
import numpy as np
# "." allows to import from the same directory
from .net1d import Net1D
#from .SCDNN_1 import (ResBlock, ResNet_EEG, SpectralConv1d)
from .SCDNN import (ResBlock, ResNet_EEG, SpectralConv1d)
from .heads import RegressionHead
from .losses import WeightedFocalLoss,GHM_Loss
import torch.nn.functional as F

#weights_num = np.array(weights)
#class_weights = torch.from_numpy(weights_num).float().to(torch.device('cuda:{}'.format(0)))
# create a specialized uning module

class SCDNN(LightningModule):
    def __init__(self,lr,n_channels=37,online_hard_mine = None, **kwargs):
        super().__init__()
        self.lr = lr
        self.online_hard_mine = online_hard_mine
        self.model = ResNet_EEG(
                    ResBlock,
                    SpectralConv1d,
                    n_channal = n_channels,
                    init_threshold = 0.2,
                    num_classes=1 #hard target
        )



    def forward(self, x):
        x = self.model(x)
        
        return x


    def training_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        device = 'cuda:0'
        logits = self.forward(x)
        #loss_function = BCELoss()
        loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        if self.online_hard_mine is not None:
            mask = hard_negative_mining(loss, y)
            # Compute final loss using only hard negatives and all positives
            loss = loss[mask].mean()
        self.log('train_loss', loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        device = 'cuda:0'
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        #loss_function = BCELoss()
        loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        out = self.forward(signals)
        #pred = self.forward(signals)

        # compute and log loss
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

class FineTuning(LightningModule):
    def __init__(self,lr,n_channels=37,online_hard_mine = None, **kwargs):
        super().__init__()
        self.lr = lr
        self.online_hard_mine = online_hard_mine
        self.model = Net1D(
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



    def forward(self, x):
        x = self.model(x)
        return x


    def training_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        device = 'cuda:0'
        logits = self.forward(x)
        loss_function = BCELoss()
        #loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        if self.online_hard_mine is not None:
            mask = hard_negative_mining(loss, y)
            # Compute final loss using only hard negatives and all positives
            loss = loss[mask].mean()
        self.log('train_loss', loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        device = 'cuda:0'
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        loss_function = BCELoss()
        #loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        out = self.forward(signals)
        #pred = self.forward(signals)

        # compute and log loss
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

def hard_negative_mining(loss, labels, neg_pos_ratio=2):
    """
    Selects hard negatives based on the loss values.
    Args:
    - loss: tensor of shape [batch_size] containing losses of each sample
    - labels: tensor containing labels of each sample
    - neg_pos_ratio: ratio of negatives to positives to keep
    Returns:
    - mask: tensor of shape [batch_size] with 1 for samples to keep, 0 to discard
    """
    pos_mask = labels > 0  # Positive samples mask
    num_pos = pos_mask.sum().item()  # Number of positive samples
    num_neg = num_pos * neg_pos_ratio  # Number of negatives to keep

    # Sort losses of negative samples
    loss_neg = loss.clone()
    loss_neg[pos_mask] = 0
    _, indices = loss_neg.sort(descending=True)
    _, order = indices.sort()

    neg_mask = order < num_neg  # Mask for hard negatives

    return pos_mask | neg_mask  # Combine pos_mask and neg_mask