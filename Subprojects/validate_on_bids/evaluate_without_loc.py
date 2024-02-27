import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df_pos = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/Models/generalized_all_ref_loc/results_localized_Hash.csv')
df_pos['Spike']=1
df_neg = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/Models/generalized_all_ref_loc/results_localized_BonoboLocal.csv')
df_neg = df_neg[df_neg.fraction_of_yes==0]
df_neg['Spike']=0
df = pd.concat([df_pos[['event_file','pred','Spike','ChannelLocation']],df_neg[['event_file','pred','Spike','ChannelLocation']]])
df = df[df.ChannelLocation=='frontal']        

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

fig.suptitle('Bids extracted spikes')
axs[0].set_ylabel('Normalized count')
axs[0].set_xlabel('Prediction')
axs[1].set_ylabel('Tpr')
axs[1].set_xlabel('Fpr')
plt.legend(frameon=False)
fig.tight_layout()
plt.savefig('quickval.png')
plt.show(block=True)