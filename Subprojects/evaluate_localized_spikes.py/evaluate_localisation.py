import argparse
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from cycler import cycler
line_cycler   = (cycler(color=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
plt.rc("axes", prop_cycle=line_cycler)

def init_locations(dataset):
    SpikeLocations = ['frontal','central','parietal', 'occipital','temporal']
    if dataset =='Loc':
        SpikeLocations +=['general']
    ChannelLocations = ['frontal','central', 'parietal', 'occipital','temporal']
    return SpikeLocations,ChannelLocations

def get_preds_and_labels(df,SpikeLocation,ChannelLocation):
    # filter for positive and negative samples
    filter_SpikeLocation = (df.SpikeLocation == SpikeLocation) 
    filter_ChannelLocation = (df.ChannelLocation==ChannelLocation)
    pos = df[filter_SpikeLocation & filter_ChannelLocation]
    neg = df[(df.label==0)&(df.ChannelLocation==ChannelLocation)][:len(pos)]
    df_eval = pd.concat([pos,neg])    
    labels = df_eval.label.round(0).astype(int)
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
    ax.plot(fpr,tpr,label=ChannelLocation+f': {roc_auc:.2f}')
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

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model',default ='Models/gen_ref_rep')
    parser.add_argument('--dataset',choices={'Loc','Clemson'})
    args = parser.parse_args()
    return args.path_model, args.dataset

if __name__ == '__main__':

    path_model, dataset = get_args()

    if dataset == 'Loc':
        df = pd.read_csv(os.path.join(path_model,'results_Loc_localized_Test.csv'))
    elif dataset == 'Clemson':
        df = pd.read_csv(os.path.join(path_model,'results_Clemson_localized_Test.csv'))
        df = df.reset_index()

    SpikeLocations,ChannelLocations = init_locations(dataset)
    results = calcualte_roc_results(df,SpikeLocations,ChannelLocations)

fig, axs = plt.subplots(2,3,figsize = (10,8),sharex=True,sharey=True)

for i,SpikeLocation in enumerate(SpikeLocations):
    ax = axs[i//3,i%3]
    results_SpikeLocation = get_results_SpikeLocation(results,SpikeLocation)
    for j, ChannelLocation in enumerate(results_SpikeLocation.ChannelLocation):
        fpr, tpr, roc_auc, N = get_results_ChannelLocation(results_SpikeLocation,ChannelLocation)
        ax=plot_subplot(ax,fpr,tpr,roc_auc,N,ChannelLocation)
    ax.set_title(f'{SpikeLocation.capitalize()} spikes, N={N}')
    ax.legend(frameon=False)
axs[0,0].set_ylabel('tpr')
axs[1,0].set_ylabel('tpr')
axs[1,2].set_aspect('equal')
for j in range(3):
    axs[1,j].set_xlabel('fpr')

fig.tight_layout()
fig.savefig(os.path.join(path_model,f'fig_general_local_{dataset}.png'))


