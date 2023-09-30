import matplotlib.pyplot as plt
import numpy as np

def learing_curve():
    mean_list = []
    std_list = []
    N = 3
    dof = 1210

    TITLE = "PPO + SM"

    for def_seed in range(N):
        mean_data = np.loadtxt('../data/dof%d/smrl_model/rl_model(sm)/%d/reward_mean.csv' % (dof, def_seed))
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

def RLvsRLinSM():
    # compare rewards

    N = 3

    o_mean_robo, o_std_robo = [], []
    b_mean_robo, b_std_robo = [], []

    for robo_i in range(1200,1220):

        our_maximum_r_policy = []
        bl_maximum_r_policy = []

        for def_seed in range(N):
            each_run_robot_r1 = np.loadtxt('../data/dof%d/smrl_model/rl_model(sm)/%d/reward_mean.csv'%(robo_i, def_seed))
            our_maximum_r_policy.append(max(each_run_robot_r1))

            each_run_robot_r2 = np.loadtxt('../data/dof%d/RL_model/%d/r.csv'%(robo_i, def_seed))
            bl_maximum_r_policy.append(max(each_run_robot_r2))

        o_mean_robo.append(np.mean(our_maximum_r_policy))
        o_std_robo.append(np.std(our_maximum_r_policy))

        b_mean_robo.append(np.mean(bl_maximum_r_policy))
        b_std_robo.append(  np.std(bl_maximum_r_policy))

    print(b_mean_robo)
    print(b_std_robo)


    print(o_mean_robo)
    print(o_std_robo)
RLvsRLinSM()

