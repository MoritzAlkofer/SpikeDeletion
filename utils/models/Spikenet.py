import torch
import torch.nn as nn
import pytorch_lightning as pl

'''

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 128, 1, 37)]      0         
                                                                 
 permute_1 (Permute)         (None, 128, 37, 1)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 37, 32)       256       
                                                                 
 conv2d_2 (Conv2D)           (None, 128, 37, 32)       8192      
                                                                 
 batch_normalization_1 (Batc  (None, 128, 37, 32)      128       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 128, 37, 32)       0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 32, 37, 32)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 32, 37, 32)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 32, 1, 32)         37888     
                                                                 
 conv2d_4 (Conv2D)           (None, 32, 1, 64)         16384     
                                                                 
 batch_normalization_2 (Batc  (None, 32, 1, 64)        256       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 32, 1, 64)         0         
                                                                 
 dropout_2 (Dropout)         (None, 32, 1, 64)         0         
                                                                 
 conv2d_5 (Conv2D)           (None, 32, 1, 64)         32768     
                                                                 
 batch_normalization_3 (Batc  (None, 32, 1, 64)        256       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_3 (LeakyReLU)   (None, 32, 1, 64)         0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 8, 1, 64)         0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 8, 1, 64)          0         
                                                                 
 conv2d_6 (Conv2D)           (None, 8, 1, 96)          49152     
                                                                 
 batch_normalization_4 (Batc  (None, 8, 1, 96)         384       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_4 (LeakyReLU)   (None, 8, 1, 96)          0         
                                                                 
 dropout_4 (Dropout)         (None, 8, 1, 96)          0         
                                                                 
 conv2d_7 (Conv2D)           (None, 8, 1, 96)          73728     
                                                                 
 batch_normalization_5 (Batc  (None, 8, 1, 96)         384       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_5 (LeakyReLU)   (None, 8, 1, 96)          0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 2, 1, 96)         0         
 2D)                                                             
                                                                 
 dropout_5 (Dropout)         (None, 2, 1, 96)          0         
                                                                 
 conv2d_8 (Conv2D)           (None, 2, 1, 128)         98304     
                                                                 
 batch_normalization_6 (Batc  (None, 2, 1, 128)        512       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_6 (LeakyReLU)   (None, 2, 1, 128)         0         
                                                                 
 dropout_6 (Dropout)         (None, 2, 1, 128)         0         
                                                                 
 reshape_1 (Reshape)         (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
 activation_1 (Activation)   (None, 1)                 0         
                                                                 
'''

class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size, spatial_kernel_size):
        super().__init__()
        self.temporal_conv = nn.Conv1d(in_channels, in_channels, kernel_size=temporal_kernel_size, groups=in_channels,padding=1)
        self.spatial_conv = nn.Conv1d(in_channels, out_channels, kernel_size=spatial_kernel_size,padding=1)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels),
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SpikeNet(pl.LightningModule):
    def __init__(self, num_classes=1):
        super().__init__()
        self.factorized_conv = FactorizedConv(1, 1, temporal_kernel_size=3, spatial_kernel_size=3)
        self.layers = self._make_layers(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 2, num_classes)
    
    def _make_layers(self, in_channels):
        layers = []
        channel_increment = 32
        for i in range(0, 12):
            out_channels = in_channels + (i // 4) * channel_increment
            stride = 2 if i % 2 == 0 else 1
            layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.factorized_conv(x)
        print(x.shape)
        x = self.layers(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))
        return x

x = torch.ones((20,1,128))
net = SpikeNet()

x = net.factorized_conv(x)
x = net.layers(x)
x.shape



total_params = sum(p.numel() for p in net.parameters())