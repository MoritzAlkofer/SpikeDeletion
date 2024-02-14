import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
path_model = '../Models/generalized_referential/'
df = pd.read_csv(os.path.join(path_model,'results_deleted.csv'))

fig = plt.figure()
x = np.arange(0,20)
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
plt.errorbar(x,y,yerr,label='generalized')
print('>>> the specialized AUCs are hardcoded <<<')
plt.scatter(19,0.965,label='all bipolar')
plt.scatter(6,0.933,label='six bipolar')
plt.scatter(19,0.931,label='all referential')
plt.scatter(6,0.910,label='six referential')
plt.scatter(2,0.899,label='two referential')
yerr
plt.grid()
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))

plt.xlabel('n referential channels retained')
plt.ylabel('AUROC')
plt.ylim((0.49,1))
plt.legend()
fig.savefig(os.path.join(path_model,'non_localized_task.png'))
