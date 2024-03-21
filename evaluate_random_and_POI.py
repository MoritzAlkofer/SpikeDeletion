import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import os
from sklearn.metrics import roc_curve, auc
from local_utils import points_of_interest

from cycler import cycler
line_cycler   = (cycler(color=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
plt.rc("axes", prop_cycle=line_cycler)

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    args = parser.parse_args()
    return args.path_model

path_model = get_args()
df = pd.read_csv(os.path.join(path_model,'results_deleted.csv'))

fig = plt.figure()

x = np.arange(0,df.n_keeper.nunique())
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
auc_auc = auc(x/len(x),y)

plt.errorbar(x,y,yerr,label=f'random, normed AUC={np.round(auc_auc,2)}')

print('>>> the specialized AUCs are hardcoded <<<')
# yerr
plt.grid(alpha=0.3)
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))


plt.xlabel('n channels retained')
plt.ylabel('AUROC')
plt.ylim((0.49,1))

df = pd.read_csv(os.path.join(path_model,'/results_Rep_point_of_interest_Test.csv'))
for Poi, channels in zip(points_of_interest.keys(), points_of_interest.values()):
    label = df[df.ChannelLocation==Poi].label.round(0).astype(int).to_list()
    pred = df[df.ChannelLocation==Poi].pred.to_list()
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    plt.scatter(len(channels),roc_auc,label=Poi)


plt.legend(frameon=False,loc='lower right')


fig.savefig(os.path.join(path_model,'non_localized_task.png'))
