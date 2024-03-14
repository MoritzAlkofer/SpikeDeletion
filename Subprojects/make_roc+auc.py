import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/moritz/Desktop/programming/SpikeDeletion_clean/utils')
from utils import points_of_interest

from cycler import cycler
colors=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
# Create a cycler that cycles every two times for the color
line_cycler = cycler(color=colors)
plt.rc("axes", prop_cycle=line_cycler)

def calculate_ro_curve(y_true,y_score):
    # calculate the roc curve for the model
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # calculate the AUC
    AUC = auc(fpr, tpr)
    return AUC,fpr,tpr, thresholds

def calculate_pr_curve(y_true,y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    AUC = auc(recall,precision)
    return AUC, precision, recall

def plot_ro_curve(ax,AUC,fpr,tpr,model_name):
    # workaround for tresholds are too high
    ax.plot(fpr,tpr,label=f'{model_name}: {AUC:.2f}')
    ax.plot([0,1], [0, 1], 'k',linewidth=0.3,alpha=0.3)
    ax.set_ylabel('Tpr')
    ax.set_xlabel('Fpr')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    thresholds
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

def plot_operating_point(ax,fpr,tpr,thresholds,operating_point):
    for x, y, threshold in zip(fpr,tpr,thresholds):
        if threshold<operating_point:
            break
    ax.scatter([x],[y])
    return ax

if __name__=='__main__':
    save = True
    fig,axs = plt.subplots(1,2,figsize=(10,5))

    path = 'Models/Representative/results_Rep_point_of_interest_Test.csv'
    df = pd.read_csv(path)
    
    operating_points = {'Fp1, Fp2':0.56,'C3, C4':0.55,'T3, F7, T4 F8':0.48,'T3, P3, Pz, T4, P4':0.45,'F3, C3, O1, F4, C4, O2':0.52,'all 10-20 channels':0.48}


    for ChannelLocation in points_of_interest.keys():
        # get predictions
        label = df[df.ChannelLocation==ChannelLocation].label.round(0).astype(int).to_list()
        pred = df[df.ChannelLocation==ChannelLocation].pred.to_list()
        AUC,fpr,tpr,thresholds = calculate_ro_curve(label,pred)
        axs[0] = plot_ro_curve(axs[0],AUC,fpr,tpr,ChannelLocation)
        axs[0] = plot_operating_point(axs[0],fpr,tpr,thresholds,operating_points[ChannelLocation])
        AUC, precision, recall = calculate_pr_curve(label,pred)
        axs[1] = plot_pr_curve(axs[1],AUC,precision,recall,ChannelLocation)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('2')
    fig.tight_layout()
    fig.savefig('roc_prc.png')

