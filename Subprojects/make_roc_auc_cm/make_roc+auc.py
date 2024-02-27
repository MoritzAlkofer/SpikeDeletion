import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from cycler import cycler
line_cycler   = (cycler(color=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
plt.rc("axes", prop_cycle=line_cycler)

def apply_filters(df):
    pos = df[(df['fraction_of_yes'] >= 7/8) & (df.total_votes_received >=8)]
    neg = df[(df.fraction_of_yes==0)]
    N = min(len(pos),len(neg))
    df = pd.concat([pos[:N],neg[:N]])    
    return df

def get_label_and_pred(df):
    # extract labels and predictions as lists
    labels = df.fraction_of_yes.round(0).astype(int)
    preds = df.preds.values
    return labels,preds

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

    path_models = '../../Models/'
    
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    models = ['specialized_all_bipolar','specialized_six_bipolar','specialized_all_referential','specialized_six_referential','specialized_two_referential']

    for i,model in enumerate(models):
        # get predictions
        df = pd.read_csv(os.path.join(path_models,model,'pred.csv'))
        df = apply_filters(df)
        y_true,y_score = get_label_and_pred(df)
        model_name = model.replace('_',' ').replace('specialized','')[1:] #[1:] neccessary to remove whitespace
        AUC,fpr,tpr = calculate_ro_curve(y_true,y_score)
        axs[0] = plot_ro_curve(axs[0],AUC,fpr,tpr,model_name.capitalize())
        AUC, precision, recall = calculate_pr_curve(y_true,y_score)
        axs[1] = plot_pr_curve(axs[1],AUC,precision,recall,model_name.capitalize())
    axs[0].set_aspect('equal')
    axs[1].set_aspect('2')
    fig.tight_layout()
    fig.savefig('roc_prc.png')

