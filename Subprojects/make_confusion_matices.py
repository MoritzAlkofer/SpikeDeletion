import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import points_of_interest

def get_label_and_pred(df,ChannelLocation):
    # extract labels and predictions as lists
    label = df[df.ChannelLocation==ChannelLocation].label.round(0).astype(int).to_list()
    pred = df[df.ChannelLocation==ChannelLocation].pred.to_list()
    return label,pred

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
    ax.plot(fpr,tpr,label=f'{model_name} AUC: {AUC:.3f}')
    ax.plot([0,1], [0, 1], linestyle='--')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_aspect('equal')
    ax.legend(loc="lower right")
    ax.set_title('Receiver operating characteristic curve')
    return ax

def plot_pr_curve(ax,AUC,precision,recall,model_name):
    # workaround for tresholds are too high
    ax.plot(recall,precision,label=f'{model_name} AUC: {AUC:.3f}')
    ax.plot([0,1], [0.5, 0.5], linestyle='--')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.legend(loc="lower left")
    ax.set_title('Precision recall curve')
    return ax
    
def calculate_metrics(tn,fp,fn,tp):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp) 
    precision = tp / (tp + fp)
    return accuracy.round(2), sensitivity.round(2), specificity.round(2), precision.round(2)

def plot_confusion_matrix(ax,cm,model_name):
    sns.heatmap(cm, annot=True, ax=ax,cbar=False,fmt='g')
    ax.set_title(model_name)
    ax.set_aspect('equal')
    return ax

def create_cm(y_true,y_score):
    cm = confusion_matrix(y_true, y_score)
    tn,fp,fn,tp = cm.ravel()
    accuracy, sensitivity, specificity, precision = calculate_metrics(tn,fp,fn,tp)
    return cm,accuracy, sensitivity, specificity, precision

def binarize(x,threshold):
    x = np.array(x)
    x[x>threshold] = 1
    x[x<=threshold] = 0
    return list(x)

if __name__=='__main__':
    save = True
    
    fig,axs = plt.subplots(1,6,figsize=(12,3))
    
    path = 'Models/Representative/results_Rep_point_of_interest_Test.csv'
    df = pd.read_csv(path)
    
    thresholds = {'Fp1, Fp2':0.56,'C3, C4':0.55,'T3, F7, T4 F8':0.48,'T3, P3, Pz, T4, P4':0.45,'F3, C3, O1, F4, C4, O2':0.52,'all 10-20 channels':0.48}
    results = {'Channels':[],'Accuracy':[],'Sensitivity':[],'Specificity':[],'Precision':[],'AUROC':[]}
    for i,ChannelLocation in enumerate(points_of_interest.keys()):
        # get predictions
        y_true,y_score = get_label_and_pred(df,ChannelLocation)
        auroc, _,_ = calculate_ro_curve(y_true,y_score)
        results['AUROC'].append(auroc)
        y_score = binarize(y_score,thresholds[ChannelLocation])
        cm,accuracy, sensitivity, specfificity, precision = create_cm(y_true,y_score)
        axs[i] = plot_confusion_matrix(axs[i],cm,ChannelLocation)
        results['Channels'].append(ChannelLocation)
        results['Accuracy'].append(accuracy)
        results['Sensitivity'].append(sensitivity)
        results['Specificity'].append(specfificity)
        results['Precision'].append(precision)

    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[3].set_yticks([])
    axs[4].set_yticks([])
    axs[5].set_yticks([])

    fig.tight_layout()
    fig.savefig('confusion_matix.png')

    results = pd.DataFrame(results)
    results.to_csv('metrics_table.csv',index=False)
    print(results)