import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import os

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

plt.scatter(2,0.837,label='Fp1, Fp2')
plt.scatter(2,0.918,label='C3, C4')
plt.scatter(4,0.902,label='T3, F7, T4, F8')
plt.scatter(5,0.940,label='T3, P3, Pz, T4 P4')
plt.scatter(6,0.925,label='F3, C3, O1, F4, C4, O2')
plt.scatter(19,0.941,label='All 10-20 channels')

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
plt.legend(frameon=False,loc='lower right')
fig.savefig(os.path.join(path_model,'non_localized_task.png'))
