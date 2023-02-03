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

random.seed(2022)
np.random.seed(2022)

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
    try:
        os.mkdir(log_path + "/model")
    except:
        pass
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


def collect_sm_data(step_num, step_each_epoch=6):
    S, A, NS = [], [], []
    obs = env.reset()

    for step in range(step_num):
        a = np.random.uniform(-1, 1, size=16)

        obs_, r, done, _ = env.step(a)
        A.append(a)
        S.append(obs)
        NS.append(obs_)
        obs = np.copy(obs_)

        if done or ((step + 1) % step_each_epoch == 0):
            obs = env.reset()
            print(done, step)

    S, A, NS = np.array(S), np.array(A), np.array(NS)

    np.savetxt(log_path + '/S.csv', S)
    np.savetxt(log_path + '/A.csv', A)
    np.savetxt(log_path + '/NS.csv', NS)


class SAS_data(Dataset):
    def __init__(self, mode, num_data):
        self.root = log_path
        if mode == "train":
            self.start_idx = 0
            self.end_idx = int(num_data * 0.8)
        else:
            self.start_idx = int(num_data * 0.8)
            self.end_idx = num_data

        self.all_S = np.loadtxt(log_path + 'S.csv')
        self.all_A = np.loadtxt(log_path + 'A.csv')
        self.all_NS = np.loadtxt(log_path + 'NS.csv')

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


def train_sm(epochs=1000, step_num=20000):
    batchsize = 8
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
                loss = sm_model.loss(pred_NS, NS[:, -18:])
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


def test_sm(sm_model, env, log_path, Action_array=None, TASK='f', eval_epoch_num=5, step_num=6, num_traj=25600):
    sm_model.eval()
    log_action = []
    log_gt = []
    log_pred = []
    rob_pos_ori = []
    log_results = []
    log_abc = []
    log_y = []

    y_values = []
    for epoch in range(eval_epoch_num):
        real_reward = 0

        cur_p = [0] * 3
        cur_theta = 0

        obs = env.reset()
        a = random.uniform(0, 5)
        b = random.uniform(0, 5)
        c = random.uniform(0, 5)
        log_abc.append([a, b, c])
        pos_log = 0
        # a,b,c =3.9689855041285793, 3.336564139041523, 2.2297661210501394
        a, b, c = 3, 1, 0.5
        for step in range(step_num):
            # scalse_noise = np.asarray([para_space] * num_traj).reshape((num_traj, 16))

            if Action_array == None:
                A_array_numpy = np.random.uniform(-1, 1, size=(num_traj, 16))

            S_array = np.asarray([obs] * num_traj)
            S_array = torch.from_numpy(S_array.astype(np.float32)).to(device)
            A_array = torch.from_numpy(A_array_numpy.astype(np.float32)).to(device)
            pred_ns = sm_model.forward(S_array, A_array)
            pred_ns_numpy = pred_ns[0].cpu().detach().numpy()

            # Define Object Function to Compute Rewards

            if TASK == "f":

                all_a_rewards = a * pred_ns_numpy[:, 1] - b * abs(pred_ns_numpy[:, 5]) - c * abs(pred_ns_numpy[:, 0])
            elif TASK == "l":
                all_a_rewards = pred_ns_numpy[:,
                                5]  # - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
            elif TASK == "r":
                all_a_rewards = -pred_ns_numpy[:,
                                 5]  # - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
            elif TASK == "stop":
                all_a_rewards = 10 - abs(pred_ns_numpy[:, 1]) - abs(pred_ns_numpy[:, 0])
            elif TASK == "move_r":
                all_a_rewards = pred_ns_numpy[:, 0]
            elif TASK == "b":
                all_a_rewards = -10 * pred_ns_numpy[:, 1] - b * abs(pred_ns_numpy[:, 5]) - c * abs(pred_ns_numpy[:, 0])
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
            obs, r, done, _ = env.step(choose_a)
            log_action.append(choose_a)

            real_reward += r
            gt = np.copy(obs[-18:])

            log_gt.append(gt)
            log_pred.append(pred)
            pos_log, ori_log = env.robot_location()
            print(pos_log, ori_log, a, b, c)
            rob_pos_ori.append(np.concatenate((ori_log, pos_log)))

        log_y.append(pos_log[1])
        log_results.append(real_reward)
        pos, ori = env.robot_location()
        y_values.append(pos[1])

    print(np.max(log_y), log_abc[np.argmax(log_y)])
    np.savetxt(log_path + '/pred.csv', np.asarray(log_pred))
    np.savetxt(log_path + '/gt.csv', np.asarray(log_gt))
    np.savetxt(log_path + '/rob_pos_ori.csv', np.asarray(rob_pos_ori))
    np.savetxt(log_path + '/log_action.csv', np.asarray(log_action))

    result_mean = np.mean(log_results)
    result_std = np.std(log_results)
    print("rewards:", result_mean, result_std)
    print("Y:", np.mean(y_values), np.std(y_values))


def train_agent_with_sm(env, model, log_path):
    # Random Agent, before training
    mean_reward_before_train, std_reward_before_train = evaluate_RL(env, model)
    r_m_each_epoch = [mean_reward_before_train]
    r_s_each_epoch = [std_reward_before_train]

    epoch_num = 12800
    best_r = -np.inf

    for epoch in range(epoch_num):
        print('epoch:%d' % epoch)
        env.sm_model_world = True
        model.learn(total_timesteps=6)

        if epoch % 50 == 0 or epoch == (epoch_num - 1):
            env.sm_model_world = False
            mean_reward, std_reward = evaluate_RL(env, model)

            if best_r < mean_reward:
                best_r = mean_reward
                model.save(log_path + "/best_model")

            r_m_each_epoch.append(mean_reward)
            r_s_each_epoch.append(std_reward)

            np.savetxt(log_path + "/reward_mean.csv", np.asarray(r_m_each_epoch))
            np.savetxt(log_path + "/reward_std.csv", np.asarray(r_s_each_epoch))
            model.save(log_path + "/model%d" % (epoch))

    np.savetxt(log_path + "/reward_mean.csv", np.asarray(r_m_each_epoch))
    np.savetxt(log_path + "/reward_std.csv", np.asarray(r_s_each_epoch))
    plt.errorbar(range(len(r_m_each_epoch)), r_m_each_epoch, r_s_each_epoch)
    plt.savefig(log_path + "rewards_plot.png")


if __name__ == '__main__':

    name = "V000"
    sm_model = FastNN(18 + 16, 18)  # ,activation_fun="Relu"
    # RL training
    # Train_flag = True
    Train_flag = False
    p.connect(p.DIRECT)

    rl_all_dof_data = []
    for dof in dof_list:
        print("DOF", dof)
        random.seed(2022)
        np.random.seed(2022)
        log_path = '../data/dof%d/' % (dof)
        inital_para = np.loadtxt("../controller100/control_para/para.csv")
        para_space = np.loadtxt("../controller100/control_para/dof%d/para_range.csv" % dof)

        os.makedirs(log_path, exist_ok=True)

        env = OSM_Env(dof, inital_para, para_space,
                      data_save_pth=log_path,
                      urdf_path="../CAD2URDF/V000/urdf/V000.urdf")

        sub_logger_r = []
        sub_logger_y = []

        for sub_process in range(3):
            print('sub_process', sub_process)
            torch.manual_seed(sub_process)
            log_path_ = log_path + '/rl_model/%d/' % sub_process
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
