from bl3_env import OSM_Env
from stable_baselines3 import PPO, SAC, DDPG
from model import *
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from controller100.control import *
import pybullet as p



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)


def evaluate_RL(eval_env, model, steps_each_episode=6):
    eval_env.sm_model_world = False
    print("Evaluation: sm_model_world ", eval_env.sm_model_world)

    episode_rewards = 0
    obs = eval_env.reset()
    action_list = []
    for i in range(steps_each_episode):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = eval_env.step(action)
        action_list.append(action)
        episode_rewards += r
        if done:
            print("fail")
            break

    pos, ori = eval_env.robot_location()
    print("rewards", episode_rewards)
    print("Y", pos[1])

    return episode_rewards, pos[1]


def train_agent(log_path_, epoch_num=6400, num_step_for_eval=300):
    # Random Agent, before training
    r, y = evaluate_RL(env, model)
    r_each_epoch = [r]
    y_each_epoch = [y]

    best_r = -np.inf

    for epoch in range(epoch_num):
        print('epoch:%d' % epoch)

        model.learn(total_timesteps=6)

        if (epoch + 1) % 50 == 0:
            r, y = evaluate_RL(env, model)

            if best_r < r:
                best_r = r
                model.save(log_path_ + "/model/best_model")

            r_each_epoch.append(r)
            print(r, env.count)

            np.savetxt(log_path_ + "/r.csv", np.asarray(r_each_epoch))
            np.savetxt(log_path_ + "/y.csv", np.asarray(y_each_epoch))
            print(log_path_ + "/model/model%d" % (epoch + 1))
            model.save(log_path_ + "/model/model%d" % (epoch + 1))

    np.savetxt(log_path_ + "/r.csv", np.asarray(r_each_epoch))
    np.savetxt(log_path_ + "/y.csv", np.asarray(r_each_epoch))




if __name__ == '__main__':

    name = "V000"
    sm_model = FastNN(18 + 16, 18)  # ,activation_fun="Relu"
    # RL training
    # Train_flag = True
    Train_flag = False
    p.connect(p.GUI)
    rl_all_dof_data = []

    mode = 0
    if mode == 0:
        start_id = 1230
        for dof in dof_list:
            print("DOF", dof)

            log_path = '../data/dof%d/' % (dof)
            os.makedirs(log_path,exist_ok=True)
            inital_para = np.loadtxt("../controller100/control_para/para.csv")

            if dof >1200:
                # para_space = np.loadtxt("../controller100/control_para/para_range.csv")
                # para_space = np.random.normal(para_space, scale=0.1)
                # para_space = np.clip(para_space, 0, 1)
                # np.savetxt(log_path + '/para_range.csv', para_space)
                para_space = np.loadtxt(log_path + "para_range.csv")

            else:
                para_space = np.loadtxt("../controller100/control_para/para_range.csv")


            random.seed(2022)
            np.random.seed(2022)

            os.makedirs(log_path, exist_ok=True)

            env = OSM_Env(dof, inital_para, para_space,
                          data_save_pth=log_path,
                          urdf_path="../CAD2URDF/V000/urdf/V000.urdf")

            sub_logger_r = []
            sub_logger_y = []

            for sub_process in range(3):
                print('sub_process', sub_process)
                torch.manual_seed(sub_process)
                log_path_ = log_path + '/RL_model/%d/' % sub_process
                os.makedirs(log_path_, exist_ok=True)

                if Train_flag:
                    model = PPO("MlpPolicy", env, n_steps=6, verbose=0, batch_size=6)
                    train_agent(log_path_, epoch_num=100)
                else:
                    model = PPO.load("%s/model/best_model" % (log_path_), env)
                    r, y = evaluate_RL(env, model)
                    sub_logger_r.append(r)
                    sub_logger_y.append(y)

            if not Train_flag:
                rl_all_dof_data.append(sub_logger_r)
                np.savetxt('../paper_data/rl_baseline_logger.csv', np.asarray(rl_all_dof_data))

    if mode == 1:
        for dof in range(1200, 1220):
            print("DOF", dof)

            log_path = '../data/dof%d/' % (dof)
            os.makedirs(log_path, exist_ok=True)
            inital_para = np.loadtxt("../controller100/control_para/para.csv")

            if dof > 1200:
                # para_space = np.random.normal(para_space, scale=0.05)
                para_space = np.loadtxt(log_path + "para_range.csv")
            else:
                para_space = np.loadtxt("../controller100/control_para/para_range.csv")

            # para_space = np.clip(para_space,0,1)
            # np.savetxt(log_path + '/para_range.csv',para_space)
            random.seed(2022)
            np.random.seed(2022)

            os.makedirs(log_path, exist_ok=True)

            env = OSM_Env(dof, inital_para, para_space,
                          data_save_pth=log_path,
                          urdf_path="../CAD2URDF/V000/urdf/V000.urdf",sub_process = sub_process)

            sub_logger_r = []
            sub_logger_y = []

            for sub_process in range(3):
                print('sub_process', sub_process)
                torch.manual_seed(sub_process)
                log_path_ = log_path + '/RL_model/%d/' % sub_process
                os.makedirs(log_path_, exist_ok=True)

                if Train_flag:
                    model = PPO("MlpPolicy", env, n_steps=6, verbose=0, batch_size=6)
                    train_agent(log_path_, epoch_num=100)
                else:
                    model = PPO.load("%s/model/best_model" % (log_path_), env)
                    r, y = evaluate_RL(env, model)
                    sub_logger_r.append(r)
                    sub_logger_y.append(y)

            if not Train_flag:
                rl_all_dof_data.append(sub_logger_r)
                np.savetxt('../paper_data/rl_baseline_logger.csv', np.asarray(rl_all_dof_data))
