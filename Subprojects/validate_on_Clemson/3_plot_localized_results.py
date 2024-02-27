import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def filter_channel_location(df,channel_location):
    filter = df.ChannelLocation==channel_location
    return df[filter]

def filter_spike_loc(df,spike_location):
    filter= df[spike_location]==1
    return df[filter]

def filter_neg(df):
    return df[df.Spike==0]

def balance_spikes(pos,neg):
    N=min([len(pos),len(neg)])
    pos = pos[:N]
    neg = neg[:N]
    return pd.concat([pos,neg])

def filter_sub_df(df,channel_location,spike_location):

    df = filter_channel_location(df,channel_location)
    pos = filter_spike_loc(df,spike_location)
    neg = filter_neg(df)
    df = balance_spikes(pos,neg)
    return df





if __name__ =='__main__':
    locations = ['frontal','central', 'parietal', 'occipital','temporal']

    df = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/Models/generalized_all_ref_loc/results_localized_Clemson.csv')
    loc = pd.read_csv('spikeloc.csv')
    df =df.merge(loc[['event_file','Spike']+locations],on='event_file',how='left')


    results={'spikeLocation':[],'channelLocation':[],'fpr':[], 'tpr':[], 'roc_auc':[],'N':[]}

    for spike_location in locations:
        for channel_location in locations:

            sub_df= filter_sub_df(df,channel_location,spike_location)
            
            preds,labels = sub_df.pred.to_list(), sub_df.Spike
            fpr, tpr, thresholds = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)

            results['spikeLocation'].append(spike_location)        
            results['channelLocation'].append(channel_location)
            results['fpr'].append(fpr)
            results['tpr'].append(tpr)
            results['roc_auc'].append(roc_auc)
            results['N'].append(len(sub_df))

results = pd.DataFrame(results)
fig, axs = plt.subplots(2,3,figsize = (12,5),sharex=True,sharey=True)

for i,spike_location in enumerate(locations):
    for j,channel_location in enumerate(locations):
        filter_channelLocation = results.channelLocation==channel_location
        filter_spikeLocation = results.spikeLocation==spike_location
        
        
        fpr, tpr, roc_auc, N = results[filter_channelLocation & filter_spikeLocation].iloc[0][['fpr', 'tpr', 'roc_auc', 'N']]

        axs[i//3,i%3].plot(fpr,tpr,label=channel_location+f' ROC: {roc_auc:.2f}')
    axs[i//3,i%3].set_title(f'{spike_location}, N={N}')
    axs[i//3,i%3].legend()

fig.suptitle('Localized spike detection with localized node placement')
axs[0,0].set_ylabel('tpr')
axs[1,0].set_ylabel('tpr')
for j in range(3):
    axs[1,j].set_xlabel('fpr')

fig.tight_layout()
plt.savefig('results_clemson.png')
plt.show(block=True)