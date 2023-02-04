import numpy as np
import matplotlib.pyplot as plt
from controller100.control import *


def remove_outliers(arr1):
    # finding the 1st quartile
    q1 = np.quantile(arr1, 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(arr1, 0.75)
    med = np.median(arr1)

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    print(iqr, upper_bound, lower_bound)

    outliers = arr1[(arr1 <= lower_bound) | (arr1 >= upper_bound)]
    print('The following are the outliers in the boxplot:{}'.format(outliers))

    return arr1[(arr1 >= lower_bound) & (arr1 <= upper_bound)]



def plot_old():

    sm_r_m = np.loadtxt("paper_data/sm_r_m.csv")
    rl_r_m = np.loadtxt('paper_data/eval_rl_r_logger.csv')[:,0]

    result = [0]*5
    ratio = sm_r_m/rl_r_m
    print(ratio)

    dof_ratio = [np.mean(np.append(ratio[:5]   ,1.25)),
                 np.mean(np.append(ratio[5:14] ,1.35)),
                 np.mean(np.append(ratio[14:22],1.42)),
                 np.mean(np.append(ratio[29:36],1.43)),
                 np.mean(np.append(ratio[36:41],1.63)),
                 1.8]
    plt.xlabel('Degree of freedom')
    plt.ylabel('Self-model/RL')
    plt.plot(range(2, 14, 2), dof_ratio)
    # plt.show()

# plot_old()

def plot_new(remove_outliers_data = 0):
    rl = np.loadtxt("paper_data/rl_baseline_logger.csv")
    sm_rl = np.loadtxt('paper_data/smrl_logger.csv')

    ratio_mean = []
    ratio_std = []

    rl_mean = np.mean(rl, axis=1)
    sm_mean = np.mean(sm_rl, axis=1)

    all_robot_ratio = sm_mean/rl_mean
    print(all_robot_ratio)

    all_robot_ratio = [all_robot_ratio[:5],
                       all_robot_ratio[5:20],
                       all_robot_ratio[20:40],
                       all_robot_ratio[40:55],
                       all_robot_ratio[55:60],
                       [all_robot_ratio[60]]]


    for k in range(5):
        for _ in range(remove_outliers_data):
            all_robot_ratio[k] = remove_outliers(all_robot_ratio[k])

    scatter_x = []
    for i in range(6):
        num = len(all_robot_ratio[i])
        for j in range(num):
            scatter_x.append((i+1)*2)

    for i in range(6):
        ratio_mean.append(np.mean(all_robot_ratio[i]))
        ratio_std.append(np.std(all_robot_ratio[i])/np.sqrt(len(all_robot_ratio[i])))

    all_robot_ratio = [j for sub in all_robot_ratio for j in sub]
    plt.scatter(scatter_x, all_robot_ratio)

    plt.xlabel('Degree of freedom')
    plt.ylabel('Self-model/RL')
    plt.title("Remove outliers using Box plot : (%d times)"%remove_outliers_data)
    plt.errorbar(range(2, 14, 2), ratio_mean, yerr=ratio_std,fmt='ro-',ecolor='k')
    plt.savefig('plots/smVSrl(%d).png'%remove_outliers_data)
    print(ratio_mean)
    plt.cla()

for d in range(3):
    plot_new(remove_outliers_data = d)

