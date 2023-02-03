import numpy as np
import random

inner_motor_index = [0, 3, 6, 9]
middle_motor_index = [1, 4, 7, 10]
outer_motor_index = [2, 5, 8, 11]

dof_list = [
    200, 201, 202, 203, 204,
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414,
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,
    800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814,
    1000, 1001, 1002, 1003, 1004,
    12
]


def random_para():
    para = np.zeros(16)
    for i in range(16):
        para[i] = random.uniform(-1, 1)
    for i in range(2, 6):
        para[i] *= 2 * np.pi

    return para


def batch_random_para(para_batch):
    for i in range(16):
        para_batch[i][i] = random.uniform(-1, 1)
        if i in [2, 3, 4, 5]:
            para_batch[i][i] *= 2 * np.pi

    return para_batch


def sin_move(ti, para, sep=16):
    # print(para)
    s_action = np.zeros(12)
    # print(ti)
    s_action[0] = para[0] * np.sin(ti / sep * 2 * np.pi + para[2]) + para[10]  # left   hind
    s_action[3] = para[1] * np.sin(ti / sep * 2 * np.pi + para[3]) + para[11]  # left   front
    s_action[6] = para[1] * np.sin(ti / sep * 2 * np.pi + para[4]) - para[11]  # right  front
    s_action[9] = para[0] * np.sin(ti / sep * 2 * np.pi + para[5]) - para[10]  # right  hind

    s_action[1] = para[6] * np.sin(ti / sep * 2 * np.pi + para[2]) - para[12]  # left   hind
    s_action[4] = para[7] * np.sin(ti / sep * 2 * np.pi + para[3]) - para[13]  # left   front
    s_action[7] = para[7] * np.sin(ti / sep * 2 * np.pi + para[4]) - para[13]  # right  front
    s_action[10] = para[6] * np.sin(ti / sep * 2 * np.pi + para[5]) - para[12]  # right  hind

    s_action[2] = para[8] * np.sin(ti / sep * 2 * np.pi + para[2]) + para[14]  # left   hind
    s_action[5] = para[9] * np.sin(ti / sep * 2 * np.pi + para[3]) + para[15]  # left   front
    s_action[8] = para[9] * np.sin(ti / sep * 2 * np.pi + para[4]) + para[15]  # right  front
    s_action[11] = para[8] * np.sin(ti / sep * 2 * np.pi + para[5]) + para[14]  # right  hind

    return s_action


def change_parameters(para):
    for i in range(16):
        rdm_number = random.uniform(-1, 1)
        if random.getrandbits(1):
            if i in [0, 1, 6, 7, 8, 9]:
                para[i] = rdm_number
            elif i in range(2, 6):
                para[i] = 2 * np.pi * rdm_number
            elif i in range(10, 12):
                para[i] = rdm_number * (1 - abs(para[i - 10]))
            elif i in range(12, 16):
                para[i] = rdm_number * (1 - abs(para[i - 6]))
    return para


def dof_to_RobotMotorIndex(dof):
    #          Back
    ##    11            2
    ##       10       1
    ##          9   0
    #             @
    ##          6   3
    ##        7        4
    ##      8            5
    #          Front
    # 2  DOF: 6
    # 4  DOF: 15
    # 6  DOF: 20
    # 8  DOF: 15
    # 10 DOF: 6

    robot_type = [[3, 6],  # 206
                  [0, 3, 6, 9],  # 400
                  [0, 3, 4, 6, 7, 9],  # 603
                  [0, 1, 3, 4, 6, 7, 9, 10],  # 807
                  [0, 1, 3, 4, 5, 6, 7, 8, 9, 10],  # 1005
                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    # TO DO:
    # change 2 to 205
    # retrain 401 to 408 range(401,409)
    if dof <= 12:
        robot_actuated = robot_type[(dof - 1) // 2]
    elif dof == 200:
        robot_actuated = [0, 9]
    elif dof == 201:
        robot_actuated = [4, 7]
    elif dof == 202:
        robot_actuated = [1, 10]
    elif dof == 203:
        robot_actuated = [5, 8]
    elif dof == 204:
        robot_actuated = [2, 11]
    elif dof == 205:
        robot_actuated = [3, 6]

    elif dof == 400:
        robot_actuated = [0, 3, 6, 9]
    elif dof == 401:
        robot_actuated = [0, 4, 7, 9]
    elif dof == 402:
        robot_actuated = [0, 5, 8, 9]
    elif dof == 403:
        robot_actuated = [1, 3, 6, 10]
    elif dof == 404:
        robot_actuated = [1, 4, 7, 10]
    elif dof == 405:
        robot_actuated = [1, 5, 8, 10]
    elif dof == 406:
        robot_actuated = [2, 3, 6, 11]
    elif dof == 407:
        robot_actuated = [2, 4, 7, 11]
    elif dof == 408:
        robot_actuated = [2, 5, 8, 11]
    elif dof == 409:
        robot_actuated = [0, 1, 9, 10]
    elif dof == 410:
        robot_actuated = [0, 2, 9, 11]
    elif dof == 411:
        robot_actuated = [1, 2, 10, 11]
    elif dof == 412:
        robot_actuated = [3, 4, 6, 7]
    elif dof == 413:
        robot_actuated = [3, 5, 6, 8]
    elif dof == 414:
        robot_actuated = [4, 5, 7, 8]


    elif dof == 600:
        robot_actuated = [0, 1, 2, 9, 10, 11]
    elif dof == 601:
        robot_actuated = [0, 1, 3, 6, 9, 10]
    elif dof == 602:
        robot_actuated = [2, 4, 5, 7, 8, 11]
    elif dof == 603:
        robot_actuated = [0, 3, 4, 6, 7, 9]
    elif dof == 604:
        robot_actuated = [0, 1, 5, 8, 9, 10]
    elif dof == 605:
        robot_actuated = [0, 2, 3, 6, 9, 11]
    elif dof == 606:
        robot_actuated = [0, 2, 4, 7, 9, 11]
    elif dof == 607:
        robot_actuated = [0, 2, 5, 8, 9, 11]
    elif dof == 608:
        robot_actuated = [1, 2, 3, 6, 10, 11]
    elif dof == 609:
        robot_actuated = [0, 3, 5, 6, 8, 9]
    elif dof == 610:
        robot_actuated = [0, 4, 5, 7, 8, 9]
    elif dof == 611:
        robot_actuated = [3, 4, 5, 6, 7, 8]
    elif dof == 612:
        robot_actuated = [1, 3, 4, 6, 7, 10]
    elif dof == 613:
        robot_actuated = [2, 3, 4, 6, 7, 11]
    elif dof == 614:
        robot_actuated = [1, 3, 5, 6, 8, 10]
    elif dof == 615:
        robot_actuated = [2, 3, 5, 6, 8, 11]
    elif dof == 616:
        robot_actuated = [0, 1, 4, 7, 9, 10]
    elif dof == 617:
        robot_actuated = [1, 2, 4, 7, 10, 11]
    elif dof == 618:
        robot_actuated = [1, 2, 5, 8, 10, 11]
    elif dof == 619:
        robot_actuated = [1, 4, 5, 7, 8, 10]

    elif dof == 800:
        robot_actuated = [0, 2, 3, 5, 6, 8, 9, 11]
    elif dof == 801:
        robot_actuated = [0, 1, 2, 3, 6, 9, 10, 11]
    elif dof == 802:
        robot_actuated = [0, 1, 2, 4, 7, 9, 10, 11]
    elif dof == 803:
        robot_actuated = [0, 1, 2, 5, 8, 9, 10, 11]
    elif dof == 804:
        robot_actuated = [0, 3, 4, 5, 6, 7, 8, 9]
    elif dof == 805:
        robot_actuated = [1, 3, 4, 5, 6, 7, 8, 10]
    elif dof == 806:
        robot_actuated = [2, 3, 4, 5, 6, 7, 8, 11]
    elif dof == 807:
        robot_actuated = [0, 1, 3, 4, 6, 7, 9, 10]
    elif dof == 808:
        robot_actuated = [0, 1, 3, 5, 6, 8, 9, 10]
    elif dof == 809:
        robot_actuated = [0, 1, 4, 5, 7, 8, 9, 10]
    elif dof == 810:
        robot_actuated = [0, 2, 3, 4, 6, 7, 9, 11]
    elif dof == 811:
        robot_actuated = [0, 2, 4, 5, 7, 8, 9, 11]
    elif dof == 812:
        robot_actuated = [1, 2, 3, 4, 6, 7, 10, 11]
    elif dof == 813:
        robot_actuated = [1, 2, 3, 5, 6, 8, 10, 11]
    elif dof == 814:
        robot_actuated = [1, 2, 4, 5, 7, 8, 10, 11]


    elif dof == 1000:
        robot_actuated = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    elif dof == 1001:
        robot_actuated = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    elif dof == 1002:
        robot_actuated = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]
    elif dof == 1003:
        robot_actuated = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]
    elif dof == 1004:
        robot_actuated = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11]
    elif dof == 1005:
        robot_actuated = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

    else:
        robot_actuated = None

    return robot_actuated


if __name__ == '__main__':
    para_batch = np.array([random_para()] * 16)
    batch_random_para(para_batch)
    print(para_batch)
