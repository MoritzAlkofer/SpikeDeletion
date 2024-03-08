import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
path_model = '../Models/SpikeNet_gen_rep_aug/'
df = pd.read_csv(os.path.join(path_model,'results_deleted.csv'))

fig = plt.figure()
x = np.arange(0,20)
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
plt.errorbar(x,y,yerr,label='random')
print('>>> the specialized AUCs are hardcoded <<<')
plt.scatter(2,0.86,label='forehead channels')
plt.scatter(6,0.92,label='psg channels')
plt.scatter(19,0.944,label='all channels')
# yerr
plt.grid(alpha=0.3)
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))

plt.xlabel('n referential channels retained')
plt.ylabel('AUROC')
plt.ylim((0.49,1))
plt.legend(frameon=False)
fig.savefig(os.path.join(path_model,'non_localized_task.png'))
