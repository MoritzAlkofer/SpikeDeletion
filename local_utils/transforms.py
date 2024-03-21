import numpy as np

class ChannelFlipper():

    def __init__(self, p, channels):
        self.p = p
        # get flipped order of channel strings
        if not '-' in channels[0]:
            channels_flipped = self._monopolar_flipper(channels)
        elif 'avg' in channels[0]:
            channels_flipped = self._average_flipper(channels)
        else:
            channels_flipped = self._bipolar_flipper(channels)
        
        # convert strings to indices, this is later applied to the signal
        self.flipped_order = [channels.index(c) for c in channels_flipped]
        # print output to check
        print('used channels:    '+ str(channels))
        print('flipped channels: '+ str(channels_flipped))

    def _flip_channel(self,channel):
        '''
        this function flips a channel string, e.g. Fp1 -> Fp2, F3 -> F4, etc.
        Central channels are kept in place, e.g. Fz -> Fz
        '''
        loc = channel[-1]
        # keep central channels in place
        if loc == 'z': 
            return channel
        # flip all other channels
        else: loc = int(loc)
        if loc % 2 ==1: # +1 all uneven channels. Fp1 -> Fp2, F3 -> F4, etc.
            loc +=1
        else: # -1 all even channels. Fp2 -> Fp1, F4 -> F3, etc.
            loc -= 1
        # recompose channel string
        channel = channel[:-1] + str(loc)
        return channel

    def _monopolar_flipper(self,channels):
        ''' flip all channels in a list of monopolar channels'''
        channels_flipped = []
        for channel in channels:
            channel = self._flip_channel(channel)
            channels_flipped.append(channel)
        return channels_flipped
    
    def _average_flipper(self,channels):
        ''' flip all channels in a list of monopolar channels'''
        channels_flipped = []
        for channel in channels:
            channel = self._flip_channel(channel.replace('-avg',''))
            channels_flipped.append(channel+'-avg')
        return channels_flipped
        
    def _bipolar_flipper(self,channels):
        ''' for both channels in bipolar channel, flip them separately
        keeps zentral in place '''
        channels_flipped = []
        for bipolar_channel in channels:
            channel1, channel2 = bipolar_channel.split('-')
            channel1, channel2 = self._flip_channel(channel1), self._flip_channel(channel2)
            channels_flipped.append(channel1+'-'+channel2) # recompose bipolar channel
        return channels_flipped
    
    def __call__(self, signal):
        # if random number is smaller than p, flip channels 
        if np.random.random() < self.p:
            signal = signal[self.flipped_order,:]
        return signal
    
class RandomScaler():
    def __init__(self,scale_percent):
        self.scale_percent = scale_percent

    def __call__(self,signal):
        scale_factor = 1+np.random.uniform(-self.scale_percent, self.scale_percent)
        signal = scale_factor*signal
        return signal

class Montage():
    # this version can also convert cdac monopolar montage into mgh_psg monopolar montage
    def __init__(self,montage_channels,storage_channels,echo=False):
        
        # AVERAGE MONTAGE
        # get list of all channels that should be displayed in average montage
        avg_channels = [channel for channel in montage_channels if 'avg' in channel]
        # get ids of channels 
        self.avg_ids = np.array([storage_channels.index(channel.replace('-avg','')) for channel in avg_channels])

        # BIPOLAR MONTAGE
        # get list of all channels that should be displayed in average montage
        bipolar_channels = [channel for channel in montage_channels if ('avg' not in channel)&('-' in channel)]
        # get ids of channels 
        self.bipolar_ids = np.array([[storage_channels.index(channel.split('-')[0]), storage_channels.index(channel.split('-')[1])] for channel in bipolar_channels])
    
        # conversion
        # get list of all channels that should be displayed in average montage
        monopolar_channels = [channel for channel in montage_channels if ('avg' not in channel) and ('-' not in channel)]
        # get ids of channels 
        self.monopolar_ids = np.array([storage_channels.index(channel) for channel in monopolar_channels])
    
        if echo: print('storage channels: '+str(storage_channels))
        if echo: print('montage channels: '+str(avg_channels+bipolar_channels+monopolar_channels))

    def __call__(self,signal):
        signals = []
        # AVERAGE MONTAGE
        # get average of these signals along time axis
        if len(self.avg_ids>0):
            avg_signal = signal[self.avg_ids].mean(axis=0).squeeze()
            # substract average from original signal
            avg_montaged_signal = signal[self.avg_ids] - avg_signal
            signals.append(avg_montaged_signal)
        if len(self.bipolar_ids)>0:
            # BIPOLAR MONTAGE
            bipolar_montaged_signal = signal[self.bipolar_ids[:,0]] - signal[self.bipolar_ids[:,1]]
            signals.append(bipolar_montaged_signal)
        if len(self.monopolar_ids>0):
            # add monopolar channels
            signals.append(signal[self.monopolar_ids])

        signal = np.concatenate(signals)
        return signal
    
class AverageMontage():
    # this version can also convert cdac monopolar montage into mgh_psg monopolar montage
    def __init__(self,montage_channels,storage_channels):
        
        # AVERAGE MONTAGE
        # get list of all channels that should be displayed in average montage
        avg_channels = [channel for channel in montage_channels if 'avg' in channel]
        # get ids of channels 
        self.avg_ids = np.array([storage_channels.index(channel.replace('-avg','')) for channel in avg_channels])

    def __call__(self,signal):
        signal = signal[self.avg_ids]
        # check that there are at least two channels
        # else signal - avg(signal) = 0!
        if np.sum(np.any(signal!=0,axis=1))>=2:
            avg_signal = signal.mean(axis=0).squeeze()
            # substract average from original signal
            signal = signal - avg_signal
        return signal

class Cutter():
    def __init__(self,windowsize,max_offset,Fq):
        self.windowsize = int(windowsize*Fq)
        self.max_offset = int(max_offset*Fq)

    def __call__(self,signal):
        # get index of center
        center = signal.shape[1]//2
        # get index of window start
        start = center - (self.windowsize)//2
        # shift by up -1 or 1 x offsetsize
        start = start + int(np.random.uniform(-1, 1)*self.max_offset)
        return signal[:,start:start+self.windowsize]
        
class KeepRandomChannels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,N_channels):
        self.N_channels = N_channels
    def __call__(self,signal):
        N_keeper = np.random.randint(1,self.N_channels)
        # choose n random channels
        keeper_indices = np.random.choice(self.N_channels, N_keeper, replace=False)
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
    
class KeepNRandomChannels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,N_channels,N_keeper):
        self.N_channels = N_channels
        self.N_keeper = N_keeper
    def __call__(self,signal):
        keeper_indices = np.random.choice(self.N_channels, self.N_keeper, replace=False)
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
    
class KeepFixedChannels():
    # init with total number of channels + channels to be retained
    # use: input, signal, list of channels to be retained
    # output: zero masked signal with *channels from list retained*
    def __init__(self,montage_channels,keeper_channels):
        self.montage_channels = montage_channels
        self.keeper_channels = keeper_channels
      #   print('keeping the following channels: '+keeper_channels)
    def __call__(self,signal):
        keeper_indices = np.array([self.montage_channels.index(channel) for channel in self.keeper_channels])        
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
    
def normalize(signal):
    signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
    signal = signal - np.mean(signal,axis=1,keepdims=True)
    
    return signal

def init_standard_transforms(storage_channels,montage_channels,windowsize,windowjitter,Fs):
    if any(['avg' in f for f in montage_channels]):
        montage = AverageMontage(montage_channels,storage_channels)
    montage = Montage(montage_channels,storage_channels)
    cutter = Cutter(windowsize,windowjitter,Fs)
    transforms = [montage,cutter,normalize]
    return transforms