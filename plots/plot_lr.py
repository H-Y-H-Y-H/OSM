import matplotlib.pyplot as plt
import numpy as np

mean_list = []
std_list = []
N = 5
dof = 12
def_state = 2
TITLE = "RL: PPO, state: %d n=%d, dof=%d" % (def_state, N, dof)

for i in range(N):
    mean_data = np.loadtxt('../data/dof%d/trainRL/CYCLE_6/%d/data_100/reward_mean.csv' % (dof, def_state, i))
    mean_list.append(mean_data)

mean_all_data = np.mean(mean_list, axis=0)
std_all_data = np.std(mean_list, axis=0)

plt.title(TITLE)
plt.xlabel("Amount of Data (Epochs)")
plt.ylabel("Rewards")

X = np.asarray(range(len(mean_all_data))) * 50
plt.fill_between(X, mean_all_data - std_all_data, mean_all_data + std_all_data, alpha=0.2)
plt.plot(X, mean_all_data)
plt.show()
