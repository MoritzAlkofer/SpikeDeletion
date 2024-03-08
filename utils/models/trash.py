import pytorch_lightning as pl
import torch
import torch.nn as nn


# input shape [none, channels]
class FactorizedConv(nn.Module):
    def __init__(self,eeg_channels,out_channels):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1,out_channels=out_channels,kernel_size=(1,3),stride=1,padding=(0,1))
        self.channel_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(eeg_channels,1),stride=1)

    def forward(self,x):
        # shape [none, eeg_channels,ts]
        x = x.unsqueeze(1)
        # shape [none, 1, eeg_channels,ts]
        x = self.temporal_conv(x)
        # shape [none, out_channels, eeg_channels,ts]
        x = self.channel_conv(x)
        # shape [none, out_channels, 1,ts]
        x = x.squeeze(2)
        # shape [none, out_channels, ts]
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

class DenseNet(nn.Module):
    def __init__(self,in_channels,n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x 


class SpikeNet(pl.LightningModule):
    def __init__(self, eeg_channels):
        super().__init__()
        self.factorized_conv = FactorizedConv(eeg_channels,out_channels=32)
        self.layers = self._make_layers(in_channels=32)
        self.flatten = nn.Flatten()
        self.fc = DenseNet(in_channels=832,n_classes = 1)
    
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
        # shape [none, eeg_channels, ts]
        x = self.factorized_conv(x)
        # shape [none, channels, ts]
        x = self.layers(x)
        # shape [none, channels=416, ts=2]
        x = self.flatten(x)
        # shape [none, 832]
        x = torch.sigmoid(self.fc(x))
        # shape [none, n_classes] 
        return x
    
n_channels = 20
signal = torch.ones((10,1,n_channels,128))
model = SpikeNet(eeg_channels=20)
model(signal)

torch.ones((10,n_channels,128)).shape
