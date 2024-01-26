import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../Models/generalized/pred_n_channel.csv')

fig = plt.figure()
x = np.arange(0,20)
y = df.groupby('n_keeper').mean('AUC').AUC 
yerr = df.groupby('n_keeper').AUC.std()
plt.errorbar(x,y,yerr,label='generalized')
plt.scatter(19,0.966,label='specialized ten-twenty')
plt.scatter(6,0.948,label='specialized sleep study')
plt.scatter(2,0.937,label='specialized forehead ')
yerr
plt.grid()
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))

plt.xlabel('n channels retained')
plt.ylabel('AUROC')
plt.ylim((0.49,1))
plt.legend()
fig.savefig('../Results/non_localized_task.png')
