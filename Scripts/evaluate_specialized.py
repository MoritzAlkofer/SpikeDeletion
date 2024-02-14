import argparse
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser(description='calculate AUC for model')
    parser.add_argument('--path_model',default='Models/all_channels')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--metric', choices = ['PRC','ROC','CM'])
    args = parser.parse_args()
    return args.path_model, args.show, args.save, args.metric

def apply_filters(df):
    pos = df[(df['fraction_of_yes'] >= 7/8) & (df.total_votes_received >=8) &(df.Mode=='Test')]
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

def plot_ro_curve(AUC,fpr,tpr):
    fig = plt.figure()
    # workaround for tresholds are too high
    plt.plot(fpr,tpr,label=f'Area under the curve: {AUC:.3f}')
    plt.plot([0,1], [0, 1], linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic curve')
    return fig

def plot_pr_curve(AUC,precision,recall):
    fig = plt.figure()
    # workaround for tresholds are too high
    plt.plot(recall,precision,label=f'Area under the curve: {AUC:.3f}')
    plt.plot([0,1], [0.5, 0.5], linestyle='--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc="lower right")
    plt.title('Precision recall curve')
    return fig

def calculate_metrics(tn,fp,fn,tp):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp) 
    precision = tp / (tp + fp)
    return accuracy.round(2), sensitivity.round(2), specificity.round(2), precision.round(2)

def plot_confusion_matrix(cm):
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return fig

def create_figure(metric,y_true,y_score):
    if metric == 'ROC':
        AUC,fpr,tpr = calculate_ro_curve(y_true,y_score)
        fig = plot_ro_curve(AUC,fpr,tpr)
    elif metric =='PRC':
        AUC, precision, recall = calculate_pr_curve(y_true,y_score)
        fig = plot_pr_curve(AUC,precision,recall)
    elif metric =='CM':
        cm = confusion_matrix(y_true, y_score.round())
        tn,fp,fn,tp = cm.ravel()
        accuracy, sensitivity, specificity, precision = calculate_metrics(tn,fp,fn,tp)
        print('accuracy',accuracy ,'\nsensitivity',sensitivity,'\nspecificity',specificity,'\nprecision',precision)
        fig = plot_confusion_matrix(cm)
    return fig


if __name__=='__main__':
    path_model, show, save, metric = get_args()
    # get predictions
    df = pd.read_csv(os.path.join(path_model,'pred.csv'))
    df = apply_filters(df)
    y_true,y_score = get_label_and_pred(df)
    fig = create_figure(metric,y_true,y_score)
    if show: 
        fig.show()
        input("Press enter to close")
    if save: fig.savefig(os.path.join(path_model,f'{metric}.png'))
