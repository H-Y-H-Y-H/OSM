import matplotlib.pyplot as plt
import numpy as np


plt_data = np.loadtxt('sm_rewards.csv')

X = np.asarray(range(len(plt_data)))
plt.plot(X, plt_data)
plt.show()
