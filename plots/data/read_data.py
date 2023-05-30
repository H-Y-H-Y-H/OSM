import matplotlib.pyplot as plt
import numpy as np

plt_data = np.loadtxt('reward_data_for_paper.csv')
baseline = np.mean(plt_data[:, :3],axis=1)
sm_rl = np.mean(plt_data[:, 3:],axis=1)

all_robot_config = [2] * 5 + [4] * 15 + [6] * 20 + [8] * 15 + [10] * 5 + [12] * 20
each_dof_num = [5, 15, 20, 15, 5, 20]

print(plt_data.shape)

mean_baseline_each_dof = []
std_baseline_each_dof = []

mean_smrl_each_dof = []
std_smrl_each_dof = []

count = 0
for dof_num in each_dof_num:
    print(count,count+dof_num)
    sm_each_dof_data = sm_rl[count:count+dof_num]
    rl_each_dof_data = baseline[count:count+dof_num]

    mean_smrl_each_dof.append(np.mean(sm_each_dof_data))
    std_smrl_each_dof.append(np.std(sm_each_dof_data))

    mean_baseline_each_dof.append(np.mean(rl_each_dof_data))
    std_baseline_each_dof.append(np.std(rl_each_dof_data))
    count+=dof_num

print(mean_smrl_each_dof)
print(std_smrl_each_dof)
print(mean_baseline_each_dof)
print(std_baseline_each_dof)




