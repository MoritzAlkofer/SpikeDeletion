import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
from local_utils import points_of_interest
from local_utils import calculate_bootstraps
path_results = '/home/moritz/Desktop/programming/SpikeDeletion_clean/Models/avg_rep/results_Rep_point_of_interest_Test.csv'
df = pd.read_csv(path_results)




def evaluate_bootstrap(results,ChannelLocation,bootstrap):
    mean_AUROC = bootstrap.mean().round(3)
    std_AUROC = bootstrap.std().round(3)
    conf_interval = np.percentile(bootstrap, [2.5, 97.5]).round(3)  # 95% confidence interval
    results['ChannelLocation'].append(ChannelLocation)
    results['mean AUROC'].append(mean_AUROC)
    results['std AUROC'].append(std_AUROC)
    results['Conf interval'].append(conf_interval)
    return results

results={'ChannelLocation':[],'mean AUROC':[],'std AUROC':[],'Conf interval':[]}
for ChannelLocation in points_of_interest:
    n_bootstraps = 10000
    preds = df[df.ChannelLocation==ChannelLocation].pred.to_list()
    labels = df[df.ChannelLocation==ChannelLocation].label.round(0).to_list()
    bootstrap = calculate_boostraps(preds,labels,n_bootstraps)
    results = evaluate_bootstrap(results,ChannelLocation,bootstrap)
results = pd.DataFrame(results)

results.to_csv('bootstrapped_results.csv')