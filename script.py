import os
import shutil
from controller100.control import *


def load_smrl_data():
    src = "/Users/yuhang/Downloads/data_smrl/"
    dst = "data/"

    for i in range(len(dof_list)):
        src_sm = src + 'dof%d/sm_model/train/100data/CYCLE_6/' % dof_list[i]
        dst_sm = dst + 'dof%d/smrl_model/sm_model/' % dof_list[i]

        src_smrl = src + 'dof%d/sm_model/trainRL/CYCLE_6/' % dof_list[i]
        dst_smrl = dst + 'dof%d/smrl_model/rl_model(sm)/' % dof_list[i]

        for j in range(3):
            src1 = src_sm + '%d/model_100'%j
            src2 = src_smrl + '%d/data_100/best_model.zip'%j
            os.makedirs(dst_sm + '%d'%j, exist_ok=True)
            os.makedirs(dst_smrl + '%d'%j, exist_ok=True)

            dst1 = dst_sm + '%d/model_100'%j
            dst2 = dst_smrl + '%d/best_model.zip'%j
            # dst3 = dst_ + list_folder[j] + '/model/best_model.zip'

            shutil.copy(src1, dst1)
            shutil.copy(src2, dst2)
        # break
load_smrl_data()

def load_rl_data():
    src = "/Users/yuhang/Downloads/OSM/OSM_onlyRL/data/"
    dst = "data/"

    for i in range(len(dof_list)):
        src_ = src + 'dof%d/RL_model/rl_model/' % dof_list[i]
        dst_ = dst + 'dof%d/RL_model/' % dof_list[i]

        for j in range(3):
            src1 = src_ + '%d/model/best_model.zip' % j
            src2 = src_ + '%d/model/model50.zip' % j
            src3 = src_ + '%d/model/model100.zip' % j

            os.makedirs(dst_ + '%d/model' % j, exist_ok=True)
            dst1 = dst_ + '%d/model/best_model.zip'      %j
            dst2 = dst_ + '%d/model/model50.zip'           %j
            dst3 = dst_ + '%d/model/model100.zip'%j

            shutil.copy(src1, dst1)
            shutil.copy(src2, dst2)
            shutil.copy(src3, dst3)

            src4 = src_ + '%d/y.csv'%j
            src5 = src_ + '%d/r.csv'%j

            dst4 = dst_ + '%d/y.csv'%j
            dst5 = dst_ + '%d/r.csv'%j
            shutil.copy(src4, dst4)
            shutil.copy(src5, dst5)
        # break
# load_rl_data()


def search_usefixed_urdf_wrong_data():
    all_reward = []
    for i in range(len(dof_list)):
        reward = []
        for j in range(3):
            data_pth = 'data/dof%d/RL_model/%d/r.csv'%(dof_list[i],j)
            reward_list = np.loadtxt(data_pth)
            reward.append(reward_list)
            if reward_list[0] == reward_list[1]:
                print(dof_list[i])
            if reward_list[j] == 6:
                print(dof_list[i])
        all_reward.append(reward)
    # print(all_reward)

# search_usefixed_urdf_wrong_data()


import itertools


def compute_dof_acutated(dof):
    joint_list = [0, 1, 2, 3, 4, 5]
    possible_combination = [6, 15, 20, 15, 6]
    posibi = possible_combination[(dof // 2 - 1)]
    print(posibi)

    all_robot_in_this_dof = []
    a = itertools.combinations(joint_list, dof // 2)
    for subset in a:
        all_robot_in_this_dof.append(subset)
        subset = list(subset)
        if 0 in subset:
            subset += [9]
        if 1 in subset:
            subset += [10]
        if 2 in subset:
            subset += [11]
        if 3 in subset:
            subset += [6]
        if 4 in subset:
            subset += [7]
        if 5 in subset:
            subset += [8]
        subset = sorted(subset)
        if subset not in current_list:
            print(subset)

    print(len(all_robot_in_this_dof))


current_list = [
    [0, 9],
    [4, 7],
    [1, 10],
    [5, 8],
    [2, 11],
    [3, 6],
    [0, 3, 6, 9],
    [0, 4, 7, 9],
    [0, 5, 8, 9],
    [1, 3, 6, 10],
    [1, 4, 7, 10],
    [1, 5, 8, 10],
    [2, 3, 6, 11],
    [2, 4, 7, 11],
    [2, 5, 8, 11],
    [0, 1, 9, 10],
    [0, 2, 9, 11],
    [1, 2, 10, 11],
    [3, 4, 6, 7],
    [3, 5, 6, 8],
    [4, 5, 7, 8],
    [0, 1, 2, 9, 10, 11],
    [0, 1, 3, 6, 9, 10],
    [0, 4, 5, 7, 8, 9],
    [0, 3, 4, 6, 7, 9],
    [0, 1, 5, 8, 9, 10],
    [0, 2, 3, 6, 9, 11],
    [0, 2, 4, 7, 9, 11],
    [0, 2, 5, 8, 9, 11],
    [1, 2, 3, 6, 10, 11],
    [0, 3, 5, 6, 8, 9],
    [3, 4, 5, 6, 7, 8],
    [1, 3, 4, 6, 7, 10],
    [2, 3, 4, 6, 7, 11],
    [1, 3, 5, 6, 8, 10],
    [2, 3, 5, 6, 8, 11],
    [0, 1, 4, 7, 9, 10],
    [1, 2, 4, 7, 10, 11],
    [1, 2, 5, 8, 10, 11],
    [1, 4, 5, 7, 8, 10],
    [2, 4, 5, 7, 8, 11],
    [0, 2, 3, 5, 6, 8, 9, 11],
    [0, 1, 2, 3, 6, 9, 10, 11],
    [0, 1, 2, 4, 7, 9, 10, 11],
    [0, 1, 2, 5, 8, 9, 10, 11],
    [0, 3, 4, 5, 6, 7, 8, 9],
    [1, 3, 4, 5, 6, 7, 8, 10],
    [2, 3, 4, 5, 6, 7, 8, 11],
    [0, 1, 3, 4, 6, 7, 9, 10],
    [0, 1, 3, 5, 6, 8, 9, 10],
    [0, 1, 4, 5, 7, 8, 9, 10],
    [0, 2, 3, 4, 6, 7, 9, 11],
    [0, 2, 4, 5, 7, 8, 9, 11],
    [1, 2, 3, 4, 6, 7, 10, 11],
    [1, 2, 3, 5, 6, 8, 10, 11],
    [1, 2, 4, 5, 7, 8, 10, 11],
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 11],
    [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
    [0, 1, 2, 4, 5, 7, 8, 9, 10, 11],
    [0, 1, 2, 3, 5, 6, 8, 9, 10, 11],
    [0, 1, 2, 3, 4, 6, 7, 9, 10, 11]
]  #
# compute_dof_acutated(10)
