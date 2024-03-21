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

def plot_rmses(result,label):
    minimum = result[result.rmse==result.rmse.min()].threshold.iloc[0]
    minimum = np.round(minimum,2)
    plt.plot(result.threshold,result.rmse,label = f'{label} min {minimum}')
    plt.plot((minimum,minimum),(0.3,0.7))

def add_info():
    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('RMSE')

path = 'Models/avg_rep/results_Rep_point_of_interest_Test.csv'
df = pd.read_csv(path)
df_operating_point = pd.read_csv('Subprojects/find_operating_points/poi_operating_points.csv')

table = {'channels':[],'operating_point':[],'rmse':[]}
for channels in df.ChannelLocation.unique():
    # get labels and values
    values = df[df.ChannelLocation==channels].pred.to_list()
    labels = df[df.ChannelLocation==channels].label.round(0).to_list()
    # store results
    result = calculate_rmses(values,labels)
    # store best RMSE
    operating_point,rmse = df_operating_point.loc[df.channels==channels].iloc[0]['operating_point']
    plot_rmses(result,label=channels)
    table['channels'].append(channels),table['operating_point'].append(operating_point),table['rmse'].append(rmse)

add_info()
pd.DataFrame(table).to_csv('poi_operating_points.csv',index=False)
plt.savefig('operating_points.png')
plt.close()