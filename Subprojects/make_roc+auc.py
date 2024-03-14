import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from cycler import cycler
colors=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
colors_repeated = [color for color in colors for _ in range(2)]
# Create a cycler that cycles every two times for the color
line_cycler = cycler(color=colors_repeated)
plt.rc("axes", prop_cycle=line_cycler)

def calculate_ro_curve(y_true,y_score):
    # calculate the roc curve for the model
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # calculate the AUC
    AUC = auc(fpr, tpr)
    return AUC,fpr,tpr

def calculate_pr_curve(y_true,y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    AUC = auc(recall,precision)
    return AUC, precision, recall

def plot_ro_curve(ax,AUC,fpr,tpr,model_name):
    # workaround for tresholds are too high
    ax.plot(fpr,tpr,label=f'{model_name}: {AUC:.2f}')
    ax.plot([0,1], [0, 1], 'k--')
    ax.set_ylabel('Tpr')
    ax.set_xlabel('Fpr')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.legend(loc="lower right",frameon=False)
    ax.set_title('Receiver operating characteristic curve')
    return ax

def plot_pr_curve(ax,AUC,precision,recall,model_name):
    # workaround for tresholds are too high
    ax.plot(recall,precision,label=f'{model_name}: {AUC:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0,1])
    ax.set_ylim([0.5,1])
    ax.legend(loc="lower left",frameon=False)
    ax.set_title('Precision recall curve')
    return ax
    
if __name__=='__main__':
    save = True

    path_model = '../Models/gen_ref_rep'
    
    fig,axs = plt.subplots(1,2,figsize=(10,5))

    model_names = {'pred_Rep_two_central.csv','pred_Rep_two_frontal.csv','pred_Rep_uneeg.csv','pred_Rep_epiminder_a.csv','pred_Rep_six_referential.csv','pred_Rep_all_referential.csv'}
    
    for filename, label in info.items():
        # get predictions
        df = pd.read_csv(os.path.join(path_model,filename))
        y_true = df.label.round(0).astype(int)
        AUC,fpr,tpr = calculate_ro_curve(y_true,df.pred)
        axs[0] = plot_ro_curve(axs[0],AUC,fpr,tpr,label)
        AUC, precision, recall = calculate_pr_curve(y_true,df.pred)
        axs[1] = plot_pr_curve(axs[1],AUC,precision,recall,label)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('2')
    fig.tight_layout()
    fig.savefig('roc_prc.png')

