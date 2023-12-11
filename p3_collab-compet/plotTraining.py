import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 11})

data = pd.read_csv('log_training_p3coab.csv', header=None)

plt.plot(data[0], alpha=0.5, label='max. episode reward')
plt.plot(data[3], color='g', label='average reward 100 episodes')
plt.axhline(0.5, linestyle='--', color='k', label='goal')

solved = np.where(data[3]>0.5)
xsolved = int(solved[0][0])
print(xsolved)
print(np.max(data[3]))

plt.axvline(xsolved, linestyle='--', color='g', label='solved')

plt.legend()
plt.xlabel('# Iterations')
plt.ylabel('Return')

plt.show()