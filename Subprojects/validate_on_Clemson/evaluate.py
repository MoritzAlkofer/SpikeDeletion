import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#df = pd.read_csv('results_easy_implement.csv')
df = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/Models/specialized_all_bipolar/pred_clemson.csv')

df_label = pd.read_excel('/media/moritz/Expansion/Data/Spikes_clemson_10s/tables/segments_labels_channels_montage.xlsx')

df = df.merge(df_label[['event_file','Spike']],on='event_file')


fig,axs = plt.subplots(1,2,figsize = (6,3))

axs[0].hist(df.pred.to_list(),density=True)
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,10)
axs[0].set_aspect(1/10)


preds,labels = df.pred.to_list(), df.Spike
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

axs[1].plot(fpr,tpr,label=f' ROC: {roc_auc:.2f}')
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,1)
axs[1].set_aspect('equal')

fig.suptitle('Clemson spikes')
axs[0].set_ylabel('Normalized count')
axs[0].set_xlabel('Prediction')
axs[1].set_ylabel('Tpr')
axs[1].set_xlabel('Fpr')
plt.legend(frameon=False)
fig.tight_layout()
plt.savefig('quickval.png')
plt.show(block=True)