import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance, SpikeNetInstance

model = SpikeNetInstance()

import numpy as np

x = np.ones((19,128))


import torch.nn as nn
import torch
import pytorch_lightning as pl

class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size, spatial_kernel_size):
        super().__init__()
        self.temporal_conv = nn.Conv1d(in_channels, in_channels, kernel_size=temporal_kernel_size, groups=in_channels)
        self.spatial_conv = nn.Conv1d(in_channels, out_channels, kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return x
    
x = FactorizedConv(in_channels = 19,out_channels=32,temporal_kernel_size=)