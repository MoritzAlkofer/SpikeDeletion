import argparse
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='calculate AUC for model')
    parser.add_argument('--path_model',default='Models/all_channels')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    return args.path_model, args.show, args.save

def apply_filters(df):
    pos = df[(df['fraction_of_yes'] >= 7/8) & (df.total_votes_received >=8)]
    neg = df[(df.fraction_of_yes<=1/8)]
    N = min(len(pos),len(neg))
    df = pd.concat([pos[:N],neg[:N]])    
    return df

def get_label_and_pred(df):
    # extract labels and predictions as lists
    labels = df.fraction_of_yes.round(0).astype(int)
    preds = df.preds.values
    return labels,preds

def calculate_auc(labels,pred):
    # calculate the roc curve for the model
    fpr, tpr, thresholds = roc_curve(labels, preds)
    # calculate the AUC
    roc_auc = auc(fpr, tpr)
    return roc_auc,fpr,tpr

def plot_auc(roc_auc,fpr,tpr):
    fig = plt.figure()
    # workaround for tresholds are too high
    plt.plot(fpr,tpr,label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1], [0, 1], linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate / threshold')
    plt.legend(loc="lower right")
    plt.title('AUROC curve')
    return fig

if __name__=='__main__':
    path_model, show, save = get_args()
    # get predictions
    df = pd.read_csv(os.path.join(path_model,'pred.csv'))
    df = apply_filters(df)
    labels,preds = get_label_and_pred(df)
    roc_auc,fpr,tpr = calculate_auc(labels,preds)
    fig = plot_auc(roc_auc,fpr,tpr)

    if show: 
        fig.show()
        input("Press enter to close")
    if save: fig.savefig(os.path.join(path_model,'AUC.png'))
