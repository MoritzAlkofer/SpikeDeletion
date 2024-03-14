import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler
colors=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
colors_repeated = [color for color in colors for _ in range(2)]
# Create a cycler that cycles every two times for the color
line_cycler = cycler(color=colors_repeated)
plt.rc("axes", prop_cycle=line_cycler)


def binarize(values,threshold):
    values = np.array(values)
    values[values<threshold] = 0
    values[values!=0] =1
    return list(values)

def calculate_rmses(values,labels):
    result = {'threshold':[], 'rmse':[]}
    for threshold in tqdm(np.linspace(0,1,100)):
        y_pred = binarize(values,threshold)
        y_pred = np.array(y_pred)
        labels = np.array(labels)
        rmse = np.sqrt(sum((labels-y_pred)**2)/len(labels))
        result['threshold'].append(threshold)
        result['rmse'].append(rmse)
    result = pd.DataFrame(result)
    return result

info = ['pred_Rep_two_central.csv','pred_Rep_two_frontal.csv','pred_Rep_uneeg.csv','pred_Rep_epiminder_a.csv','pred_Rep_six_referential.csv','pred_Rep_all_referential.csv']

for channels in info:
    df = pd.read_csv(os.path.join('../Models/gen_ref_rep/',channels))
    values = df.pred.to_list()
    labels = df.label.round(0).to_list()
    result = calculate_rmses(values,labels)
    minimum = result[result.rmse==result.rmse.min()].threshold.iloc[0]
    minimum = np.round(minimum,2)
    label = channels.replace('.csv','').replace('pred_Rep_','')
    plt.plot(result.threshold,result.rmse,label = f'{label} min {minimum}')
    plt.plot((minimum,minimum),(0.3,0.7))
plt.legend()
plt.savefig('test.png')
plt.close()