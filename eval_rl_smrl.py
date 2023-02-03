import numpy as np
import matplotlib.pyplot as plt
from controller100.control import *
def plot_old():

    dof_list = [
    200, 201, 202, 203, 204,
    400, 401, 402, 403, 404, 405, 406, 407, 408,
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609,
    800, 801, 802, 803, 804, 805, 806,
    1000, 1001, 1002, 1003, 1004
    ]

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

def plot_new():
    rl = np.loadtxt("paper_data/rl_baseline_logger.csv")
    sm_rl = np.loadtxt('paper_data/smrl_logger.csv')


    ratio_mean = []
    ratio_std = []

    rl_mean = np.mean(rl, axis=1)
    sm_mean = np.mean(sm_rl, axis=1)

    # list_sm = [(0,5),(5,14),(14,20),(25,33),(33,39),(39,41)]
    list_sm = [(0,5),(5,14),(14,21),(21,29),(29,35),(35,36)]

    # rl_mean_ = np.asarray(rl_mean[14:21])
    # sm_mean_ = np.asarray(sm_mean[14:21])
    # print(sm_mean_/rl_mean_)

    all_robot_ratio = sm_mean/rl_mean

    for i in range(6):
        ratio_mean.append(np.mean(all_robot_ratio[list_sm[i][0]:list_sm[i][1]]))
        ratio_std.append(np.std(all_robot_ratio[list_sm[i][0]:list_sm[i][1]]))

    # ratio_mean = np.asarray(ratio_mean)
    # ratio_std = np.asarray(ratio_std)
    print(ratio_mean)
    plt.xlabel('Degree of freedom')
    plt.ylabel('Self-model/RL')
    # plt.plot(range(2, 14, 2), ratio_mean)
    plt.errorbar(range(2, 14, 2), ratio_mean, yerr=ratio_std,fmt='ro-',ecolor='k')
    plt.show()
    print(ratio_mean)

plot_new()