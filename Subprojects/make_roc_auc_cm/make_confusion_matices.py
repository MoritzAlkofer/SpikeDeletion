import argparse
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def apply_filters(df):
    pos = df[(df['fraction_of_yes'] >= 7/8) & (df.total_votes_received >=8)&(df.Mode=='Test')]
    neg = df[(df.fraction_of_yes==0)&(df.Mode=='Test')]
    N = min(len(pos),len(neg))
    print(N)
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
    ax.plot(fpr,tpr,label=f'{model_name} AUC: {AUC:.3f}')
    ax.plot([0,1], [0, 1], linestyle='--')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
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
    return accuracy.round(3), sensitivity.round(3), specificity.round(3), precision.round(3)

def plot_confusion_matrix(ax,cm,model_name):
    sns.heatmap(cm, annot=True, ax=ax,cbar=False,fmt='g')
    ax.set_title(model_name)
    return ax

def create_cm(y_true,y_score):
    cm = confusion_matrix(y_true, y_score)
    tn,fp,fn,tp = cm.ravel()
    accuracy, sensitivity, specificity, precision = calculate_metrics(tn,fp,fn,tp)
    return cm,accuracy, sensitivity, specificity, precision

def binarize(x,threshold):
    x[x>threshold] = 1
    x[x<=threshold] = 0
    return x

if __name__=='__main__':
    save = True
    models = ['specialized_all_bipolar','specialized_six_bipolar','specialized_all_referential','specialized_six_referential','specialized_two_referential']
    path_models = '../../Models/'
    threshold = 0.5
    fig,axs = plt.subplots(1,5,figsize=(12,3))
    
    results = {'model':[],'accuracy':[],'sensitivity':[],'specificity':[],'precision':[]}
    for i,model in enumerate(models):
        # get predictions
        df = pd.read_csv(os.path.join(path_models,model,'pred.csv'))
        df = apply_filters(df)
        y_true,y_score = get_label_and_pred(df)
        y_score = binarize(y_score,threshold)
        model_name = model.replace('_',' ').replace('specialized','')
        cm,accuracy, sensitivity, specfificity, precision = create_cm(y_true,y_score)
        axs[i] = plot_confusion_matrix(axs[i],cm,model_name)
        results['model'].append(model_name)
        results['accuracy'].append(accuracy)
        results['sensitivity'].append(sensitivity)
        results['specificity'].append(specfificity)
        results['precision'].append(precision)

    axs[1].set_yticks([])
    axs[2].set_yticks([])

    fig.tight_layout()
    fig.savefig('confusion_matix.png')

    results = pd.DataFrame(results)
    results.to_csv('metrics_table.csv',index=False)
    print(results)