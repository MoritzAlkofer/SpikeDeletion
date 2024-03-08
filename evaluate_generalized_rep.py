import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description='Train model with different montages')
    parser.add_argument('--path_model')
    args = parser.parse_args()
    return args.path_model

path_model = get_args()
df = pd.read_csv(os.path.join(path_model,'results_deleted.csv'))

fig = plt.figure()
x = np.arange(0,20)
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
plt.errorbar(x,y,yerr,label='random')
print('>>> the specialized AUCs are hardcoded <<<')
# yerr
plt.grid(alpha=0.3)
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))

plt.xlabel('n referential channels retained')
plt.ylabel('AUROC')
plt.ylim((0.49,1))
plt.legend(frameon=False)
fig.savefig(os.path.join(path_model,'non_localized_task.png'))
