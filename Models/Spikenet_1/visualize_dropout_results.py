import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('results_dropout.csv')

x = np.arange(37)
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
plt.errorbar(x,y,yerr,label='generalized')
plt.xlabel('N random channels retained')
plt.ylabel('AUROC')
plt.ylim(0.45,1)
plt.grid()
plt.title('SpikeNet V1')
plt.savefig('test.png')