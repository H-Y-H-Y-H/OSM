import numpy as np
import random
from bl3_env import OSM_Env
from stable_baselines3 import PPO, SAC, DDPG
from model import *
import torch
import pybullet as p
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import Dataset, DataLoader
import time


random.seed(2022)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)


def evaluate_RL(model, num_episodes=5, num_seq=4):
    env = model.get_env()

    all_episode_rewards = []
    for epoch in range(num_episodes):
        # print(epoch)
        episode_rewards = 0
        obs = env.reset()
        action_list = []
        for i in range(6):
            # action = best_action_file[i]
            # action_logger.append(action)
            action, _ = model.predict(obs)
            obs, real_r, done, _ = env.step(action)
            action_list.append(action[0])
            if done == True:
                print("fail")
                break
            episode_rewards += real_r[0]
        np.savetxt("%s/action_list.csv" % log_path, np.asarray(action_list))
        all_episode_rewards.append(episode_rewards)

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Std reward:", std_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward, std_episode_reward


def train_agent():
    # Random Agent, before training
    mean_reward_before_train, std_reward_before_train = evaluate_RL(model, num_episodes=3)
    r_m_each_epoch = [mean_reward_before_train]
    r_s_each_epoch = [std_reward_before_train]

    epoch_num = 313
    best_r = -np.inf

    for epoch in range(epoch_num):
        print('epoch:%d' % epoch)

        model.learn(total_timesteps=64)
        mean_reward, std_reward = evaluate_RL(model, num_episodes=3)

        if best_r < mean_reward:
            best_r = mean_reward
            model.save(log_path + "/best_model")

        r_m_each_epoch.append(mean_reward)
        r_s_each_epoch.append(std_reward)
        print(mean_reward,std_reward, env.count)

        if epoch % 50 == 0:
            np.savetxt(log_path + "/reward_mean.csv", np.asarray(r_m_each_epoch))
            np.savetxt(log_path + "/reward_std.csv", np.asarray(r_s_each_epoch))
            model.save(log_path + "/model%d" % epoch)


    np.savetxt(log_path + "/reward_mean.csv", np.asarray(r_m_each_epoch))
    np.savetxt(log_path + "/reward_std.csv", np.asarray(r_s_each_epoch))
    plt.errorbar(range(epoch_num + 1), r_m_each_epoch, r_s_each_epoch)
    plt.savefig(log_path+"rewards_plot.png")


def collect_sm_data(step_num):
    S, A, NS = [], [], []
    obs = env.reset()

    for step in range(step_num):
        a = np.random.normal(inital_para, scale = para_space)
        obs_, r, done, _ = env.step(a)
        A.append(a)
        S.append(obs)
        NS.append(obs_)
        obs = np.copy(obs_)

        if done or ((step +1)%100==0):
            obs = env.reset()
            print(done,step)

    S, A, NS = np.array(S), np.array(A), np.array(NS)
    np.savetxt(log_path + 'S%d.csv'%step_num, S)
    np.savetxt(log_path + 'A%d.csv'%step_num, A)
    np.savetxt(log_path + 'NS%d.csv'%step_num, NS)


class SAS_data(Dataset):
    def __init__(self, mode, num_data):
        self.root = log_path
        if mode == "train":
            self.start_idx = 0
            self.end_idx = int(num_data * 0.8)
        else:
            self.start_idx = int(num_data * 0.8)
            self.end_idx = num_data

        self.all_S = np.loadtxt(log_path + 'S%d.csv'%num_data)
        self.all_A = np.loadtxt(log_path + 'A%d.csv'%num_data)
        self.all_NS=np.loadtxt(log_path + 'NS%d.csv'%num_data)

        self.all_S = torch.from_numpy(self.all_S).to(device, dtype=torch.float)
        self.all_A = torch.from_numpy(self.all_A).to(device, dtype=torch.float)
        self.all_NS = torch.from_numpy(self.all_NS).to(device, dtype=torch.float)

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        S = self.all_S[idx]
        A = self.all_A[idx]
        NS = self.all_NS[idx]
        sample = {'S': S, 'A': A, "NS": NS}
        return sample

    def __len__(self):
        return (self.end_idx - self.start_idx)




def train_sm(epochs = 1000,    step_num = 20000):


    batchsize = 64
    lr = 1e-4

    sm_model.to(device)
    avg_train_L = 0
    avg_valid_L = 0
    min_loss = + np.inf
    abort_learning = 0
    decay_lr = 0
    all_train_L, all_valid_L = [], []

    training_data = SAS_data(mode="train",
                             num_data=step_num)
    validation_data = SAS_data(mode="valid",
                               num_data=step_num)

    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(validation_data, batch_size=batchsize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(sm_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    for epoch in range(epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        sm_model.train()
        for batch in train_dataloader:
            S, A, NS = batch["S"], batch["A"], batch["NS"]
            pred_NS = sm_model.forward(S, A)
            loss = sm_model.loss(pred_NS, NS[:, -18:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        sm_model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                S, A, NS = batch["S"], batch["A"], batch["NS"]
                pred_NS = sm_model.forward(S, A)
                loss = sm_model.loss(pred_NS, NS[:,-18:])
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            PATH = log_path + '/best_model.pt'
            torch.save(sm_model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
            decay_lr += 1
        scheduler.step(avg_valid_L)
        np.savetxt(log_path + "training_L.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc.png")
    plt.show()


def test_sm(TASK = 'f', eval_epoch_num = 5, step_num = 6, num_traj = 1000):
    sm_model.eval()
    log_action = []
    log_gt = []
    log_pred = []
    rob_pos_ori = []
    log_results = []
    log_abc = []
    log_y = []
    for epoch in range(eval_epoch_num):
        real_reward = 0
        reward = 0

        cur_p = [0] * 3
        cur_theta = 0

        obs = env.reset()
        a = random.uniform(0,5)
        b = random.uniform(0,5)
        c = random.uniform(0,5)
        log_abc.append([a,b,c])
        pos_log = 0
        a,b,c = 0.9739379174872226, 0.6548177435069252, 0.3592433208908874
        for step in range(step_num):
            scalse_noise = np.asarray([para_space] * num_traj).reshape((num_traj, 16))
            para_array = np.asarray([inital_para] * num_traj)
            A_array_numpy = np.random.normal(loc=para_array, scale=scalse_noise, size=None)

            S_array = np.asarray([obs] * num_traj)

            S_array = torch.from_numpy(S_array.astype(np.float32)).to(device)
            A_array = torch.from_numpy(A_array_numpy.astype(np.float32)).to(device)

            pred_ns = sm_model.forward(S_array, A_array)

            pred_ns_numpy = pred_ns[0].cpu().detach().numpy()


            # Define Object Function to Compute Rewards

            if TASK == "f":
                all_a_rewards = a * pred_ns_numpy[:, 1] - b * abs(pred_ns_numpy[:, 5]) - c * abs(pred_ns_numpy[:, 0])
            elif TASK == "l":
                all_a_rewards = pred_ns_numpy[:,5]  # - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
            elif TASK == "r":
                all_a_rewards = -pred_ns_numpy[:,5]  # - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
            elif TASK == "stop":
                all_a_rewards = 10 - abs(pred_ns_numpy[:, 1]) - abs(pred_ns_numpy[:, 0])
            elif TASK == "move_r":
                all_a_rewards = pred_ns_numpy[:, 0]
            elif TASK == "b":
                all_a_rewards = -10 * pred_ns_numpy[:, 1]
            else:
                all_a_rewards = np.zeros(num_traj)
            greedy_select = int(np.argmax(all_a_rewards))
            # prob_select = random.choices(range(num_traj), weights=all_a_rewards, k=1)
            # greedy_select= 0

            choose_a = A_array_numpy[greedy_select]
            pred = np.copy(pred_ns_numpy[greedy_select])
            c_pos, c_ori = env.robot_location()
            cur_x = c_pos[0]

            cur_theta = c_ori[2]
            # cur_x += pred[0]
            # cur_theta += pred[5]

            # Only choose the action that has the largest reward.
            obs, r_step, done, _ = env.step(choose_a)
            log_action.append(choose_a)

            real_reward += r_step
            gt = np.copy(obs[-18:])

            log_gt.append(gt)
            log_pred.append(pred)
            pos_log,ori_log = env.robot_location()
            print(ori_log, pos_log, a,b,c)
            rob_pos_ori.append(np.concatenate((ori_log, pos_log)))

            if step == step_num - 1:
                break

        log_y.append(pos_log[1])
        log_results.append(real_reward)
    print(np.max(log_y),log_abc[np.argmax(log_y)])
    np.savetxt(log_path + '/pred.csv', np.asarray(log_pred))
    np.savetxt(log_path + '/gt.csv', np.asarray(log_gt))
    np.savetxt(log_path + '/rob_pos_ori.csv', np.asarray(rob_pos_ori))
    np.savetxt(log_path + '/log_action.csv', np.asarray(log_action))

    result_mean = np.mean(log_results)
    result_std = np.std(log_results)
    print("test result:", result_mean, result_std)


if __name__ == '__main__':

    name = "V000"
    sm_model = FastNN(18, 16)
    render_flag = False
    Train_flag = False
    dof = 12
    print("DOF", dof)
    log_path = 'data/dof%d/' % dof
    inital_para = np.loadtxt("controller100/control_para/dof%d/0.csv" % dof)
    para_space = np.asarray([0.6, 0.6,
                             0.6, 0.6, 0.6, 0.6,
                             0.5, 0.5,
                             0.6, 0.6,
                             0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    try: os.mkdir(log_path)
    except: pass

    env = OSM_Env(dof, render_flag, inital_para, para_space, urdf_path="CAD2URDF/V000/urdf/V000.urdf", data_save_pth = log_path)
    env.sleep_time = 0

    mode = 4
    # RL training
    if mode == 0:
        log_path = log_path + '/rl_model/'
        try:
            os.mkdir(log_path)
        except:
            pass

        if Train_flag == True:
            model = PPO("MlpPolicy", env,n_steps=64, verbose=0)
            train_agent()
        else:
            model = PPO.load("%s/model50" % log_path, env)
            mean_reward_before_train, std_reward_before_train = evaluate_RL(model, num_episodes=5)

    # collect self-model data
    if mode == 1:
        collect_sm_data(step_num=4100)

    # Train self-model
    if mode == 2:
        log_path = log_path + '/sm_model/'
        try:
            os.mkdir(log_path)
        except:
            pass
        train_sm(step_num = 100000)

    # Test self-model
    if mode == 3:
        log_path = log_path + '/sm_model/'
        sm_model.load_state_dict(torch.load(log_path+'best_model.pt', map_location=torch.device(device)))
        sm_model.to(device)
        test_sm(TASK='f', eval_epoch_num=10)

    # Show pred plots
    if mode == 4:
        pred = np.loadtxt(log_path+'/sm_model/pred.csv')
        gt = np.loadtxt(log_path + '/sm_model/gt.csv')[:, -18:]

        num = 60
        print(np.mean((pred-gt)**2))
        for i in range(6):
            plt.plot(pred[:num, i], label='pred')
            plt.plot(gt[:num, i], label='gt')
            plt.legend()
            plt.show()



