import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from make_datamodule import datamoduleLocal, get_split_dfLocal, datamoduleClemson

def init_results_dataframe_bonobo():
    # load results
    df = pd.read_csv('../Models/generalized_all_ref_loc/results_localized.csv')
    # load location labels
    df_locations = pd.read_csv('../Data/tables/locations_13FEB24.csv')
    df_locations = df_locations.rename(columns={'location':'SpikeLocation'})
    df = df.merge(df_locations[['event_file','SpikeLocation']],on='event_file',how='left')
    return df

def init_locations():
    SpikeLocations = ['frontal','central','parietal', 'occipital','temporal','general']
    ChannelLocations = ['frontal','central', 'parietal', 'occipital','temporal']
    return SpikeLocations,ChannelLocations

def get_preds_and_labels(df,SpikeLocation,ChannelLocation):
    # filter for positive and negative samples
    filter_SpikeLocation = (df.SpikeLocation == SpikeLocation) 
    filter_ChannelLocation = (df.ChannelLocation==ChannelLocation)
    filter_agreement = (df.fraction_of_yes>=7/8)
    pos = df[filter_SpikeLocation & filter_ChannelLocation & filter_agreement]
    neg = df[(df.fraction_of_yes==0)&(df.ChannelLocation==ChannelLocation)][:len(pos)]
    df_eval = pd.concat([pos,neg])    
    labels = df_eval.fraction_of_yes.round(0).astype(int)
    preds = df_eval.pred.to_list()
    return labels, preds

def get_results_SpikeLocation(results,SpikeLocation):
    SpikeResults = results[results.SpikeLocation==SpikeLocation]
    return SpikeResults

def get_results_ChannelLocation(results_SpikeLocation,ChannelLocation):
    row = results_SpikeLocation[results_SpikeLocation.ChannelLocation==ChannelLocation].iloc[0]
    fpr, tpr, roc_auc, N= row[['fpr', 'tpr', 'roc_auc', 'N']]
    return fpr,tpr,roc_auc, N

def plot_subplot(ax,fpr,tpr,roc_auc,N,ChannelLocation):
    ax.plot(fpr,tpr,label=ChannelLocation+f' ROC: {roc_auc:.2f}')
    ax.set_aspect('equal')
    return ax
  
def calcualte_roc_results(df,SpikeLocations,ChannelLocations):
    results={'SpikeLocation':[],'ChannelLocation':[],'fpr':[], 'tpr':[], 'roc_auc':[],'N':[]}
    for SpikeLocation in SpikeLocations:
        for ChannelLocation in ChannelLocations:
            labels,preds = get_preds_and_labels(df,SpikeLocation,ChannelLocation)
            fpr, tpr, thresholds = roc_curve(labels, preds)

            roc_auc = auc(fpr, tpr)
            results['SpikeLocation'].append(SpikeLocation)        
            results['ChannelLocation'].append(ChannelLocation)
            results['fpr'].append(fpr)
            results['tpr'].append(tpr)
            results['roc_auc'].append(roc_auc)
            results['N'].append(len(labels))
    return pd.DataFrame(results)

if __name__ == '__main__':
    path_model = '../Models/SpikeNet_gen_loc'

    dataset = 'Loc'

    if dataset == 'Loc':
        df = pd.read_csv(os.path.join(path_model,'results_localized_Loc.csv'))
        df_locations = pd.read_csv('../Data/tables/locations_13FEB24.csv')
        df_locations = df_locations.rename(columns={'location':'SpikeLocation'})
        df = df.merge(df_locations[['event_file','SpikeLocation']],on='event_file',how='left')
    elif dataset == 'Clemson':
        df = pd.read_csv(os.path.join(path_model,'results_localized_Clemson.csv'))
        pos,neg = df[df.fraction_of_yes==1],df[df.fraction_of_yes==0]
        df = df.rename(columns={'location':'SpikeLocation'})
        df_locations = pd.read_csv('../Data/tables/clemson_all_locs.csv')
        #df_locations = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/tables/lut_clemson_with_loc.csv')
        df_locations = df_locations.rename(columns={'location':'SpikeLocation'})
        pos = pos.merge(df_locations,on='event_file',how='right')
        df = pd.concat([pos,neg])

    SpikeLocations,ChannelLocations = init_locations()
    results = calcualte_roc_results(df,SpikeLocations,ChannelLocations)

fig, axs = plt.subplots(2,3,figsize = (10,6),sharex=True,sharey=True)

for i,SpikeLocation in enumerate(SpikeLocations):
    ax = axs[i//3,i%3]
    results_SpikeLocation = get_results_SpikeLocation(results,SpikeLocation)
    for j, ChannelLocation in enumerate(results_SpikeLocation.ChannelLocation):
        fpr, tpr, roc_auc, N = get_results_ChannelLocation(results_SpikeLocation,ChannelLocation)
        ax=plot_subplot(ax,fpr,tpr,roc_auc,N,ChannelLocation)
    ax.set_title(f'{SpikeLocation} spikes, N={N}')
    ax.legend(frameon=False)
axs[0,0].set_ylabel('tpr')
axs[1,0].set_ylabel('tpr')
for j in range(3):
    axs[1,j].set_xlabel('fpr')

fig.tight_layout()
fig.savefig(os.path.join(path_model,f'fig_general_local_{dataset}.png'))

