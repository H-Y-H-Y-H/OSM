import os
import numpy as np
from bl3_env import *
from controller100.control import *


def hill_climber(env, para, s_path, epoch_num=5000, step_num_each_epoch=100, num_individual=32):
    best_result = - np.inf
    para_batch = [para] * num_individual
    para_batch = np.array(para_batch)
    best_para = []

    for epoch in range(epoch_num):
        print(epoch)
        # Random 16 individuals with specific parameter region.
        para_batch = batch_random_para(para_batch)
        # Random all parameters for the rest 4 robots.
        for i in range(16, num_individual):
            para_batch[i] = random_para()

        result_list = []
        logger_flag = False
        for individual in range(num_individual):
            result = 0
            para = para_batch[individual]
            # Each individual will be test 100 times.
            for test_time in range(1):
                fail = False
                obs = env.reset()
                last_action = np.zeros(12)
                for step in range(step_num_each_epoch):
                    action = sin_move(step, para)

                    # action = np.random.normal(action, 0.2)

                    action = np.clip(action, -1, 1)
                    obs, r, done, info = env.step(action)
                    robo_pos, robot_ori = env.robot_location()

                    # print(robo_pos,robot_ori)
                    # print(obs[1])
                    # result +=    obs[1]*50  \
                    #           - 2*abs(robot_ori[0]) \
                    #           - 2*abs(robot_ori[1]) \
                    #           - 5*abs(robo_pos[2]-0.1629)

                    result += obs[1] * 50 \
                              - 20 * abs(obs[3]) \
                              - 20 * abs(obs[4]) \
                              - 20 * abs(obs[0]) \
                              - 20 * abs(obs[2]) \
                              - 5 * abs(robo_pos[2] - 0.1629)

                    # result += np.sum(abs(action - last_action)) - 5*abs(robot_ori[0]) - 5*abs(robot_ori[1]) - 10*abs(robo_pos[2]-0.1629)

                    # print(np.sum(abs(action - last_action)),5*abs(robot_ori[0]),abs(robot_ori[1]),abs(robo_pos[2]-0.1629))
                    last_action = np.copy(action)

                    # result +=  -2*robo_pos[0] - abs(robo_pos[1])
                    # Check the robot position. If it is doing crazy things, abort it.
                    if env.check() == True:
                        fail = True
                        break

                    # if pos[2] < 0.1:
                    #     # penalty of the stupid gait like wriggle.
                    #     result *= 0.5

            if result > best_result:
                print(epoch, result, best_result)
                logger_flag = True
                best_result = result
                best_para = para
                np.savetxt(s_path + "/%d.csv" % epoch, para)

            if logger_flag == True:
                para_batch = np.asarray([best_para] * num_individual)
                # if best_result > 1000:
                #     print("good break")
                #     break


# def init_robot_stand_up(cam=True):
#     for i in range(1000):
#         # action_init is compute by hand to make sure the robot can stay still.
#         action_init = [0, 0.8, -0.8,
#                        0.2, -0.5, 0.5,
#                        -0.2, -0.5, 0.5,
#                        0, 0.8, -0.8]
#         init_state_para = compute_init_para(action_init)
#         action = np.asarray(action_init, dtype="float64")
#         img = robo_camera(env.robotid, 12)
#         # plt.imsave("dataset/%05d.png"%i,img)
#         obs, r, done, _ = env.step(action)


def play_back(env, para, noise, epoch_num, step_num=100):
    result = 0
    for i in range(epoch_num):
        env.reset()
        for step in range(step_num):
            action = sin_move(step, para)

            action = np.random.normal(loc=action, scale=[noise * 2,
                                                         noise,
                                                         noise,
                                                         noise * 2,
                                                         noise,
                                                         noise,
                                                         noise * 2,
                                                         noise,
                                                         noise,
                                                         noise * 2,
                                                         noise,
                                                         noise], size=None)
            # action = np.random.normal(loc=action, scale=noise, size=None)
            # action = np.ones(12) * ((-1) ** (step//20))

            action = np.clip(action, -1, 1).astype(np.float32)
            obs, r, done, info = env.step(action)
            result += obs[1]
            if done == True:
                print("Shit!!!!!!!!!!!!!!!!!!!!!!")
                break
            print(step, "current step r: ", obs[1], "accumulated_r: ", result)
            print("state:", env.robot_location())


def trajectory_optimization(env, save_path, para=None, Train=False, noise=0.0):
    # Search Parameters
    if Train:
        env.camera_capture = False
        env.robot_camera = False

        # init_robot_stand_up()
        hill_climber(env, para, save_path)
    else:
        env.render = True
        env.robot_camera = True
        play_back(env, para, noise, epoch_num=10, step_num=100)


if __name__ == '__main__':

    TRAIN = True

    DOF = 12
    num_file = 0

    if TRAIN:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)

    space = np.loadtxt("controller100/control_para/para_range.csv" )
    noise = 0
    if TRAIN:
        sin_para = np.loadtxt("traj_optim/dataset/control_para/para.csv")
    else:
        sin_para = np.loadtxt("traj_optim/dataset/control_para/para.csv")

    env = OSM_Env(dof = 12, render=True, para = sin_para,noise_para=space,data_save_pth = None,
                  robot_camera=False,
                  urdf_path="../CADandURDF/robot_repo/V000/urdf/V000.urdf")
    # rand_pos= True, rand_torque= True,rand_fiction=True)
    env.spt = 0.
    env.data_collection = True

    save_path = "traj_optim/dataset/control_para/"

    try:
        os.mkdir(save_path)
    except OSError:
        pass

    trajectory_optimization(env,
                            save_path=save_path,
                            para=sin_para,
                            noise=noise,
                            Train=TRAIN)
