from main import *
from controller100.control import *
random.seed(2022)
np.random.seed(2022)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)


def collect_dyna_sm_data(step_num, use_policy, choose_a, num_candidate=1000):
    S, A, NS, R, DONE, select_ra_ornot = [], [], [], [], [], []
    obs = env.reset()

    if use_policy == 0:
        # Random action:
        for step in range(step_num):
            a = np.random.uniform(-1, 1, size=16)

            obs_, r, done, _ = env.step(a)
            A.append(a)
            S.append(obs)
            NS.append(obs_)
            obs = np.copy(obs_)
            R.append(r)
            DONE.append(done)

            if done or ((step + 1) % NUM_EACH_CYCLE == 0):
                obs = env.reset()
                print(done, step)

    elif use_policy == 1:
        sm_model.eval()
        pred_loss = []

        # Gaussian action:
        for step in range(step_num):
            A_array0 = np.asarray([choose_a] * (num_candidate // 2))
            A_array1 = np.random.uniform(-1, 1, size=((num_candidate // 2), 16))
            A_array = np.vstack((choose_a, A_array0, A_array1))
            A_array_numpy = np.random.normal(A_array, 0.2)
            S_array = np.asarray([obs] * (num_candidate + 1))
            S_array = torch.from_numpy(S_array.astype(np.float32)).to(device)
            A_array = torch.from_numpy(A_array_numpy.astype(np.float32)).to(device)

            pred_ns = sm_model.forward(S_array, A_array)
            pred_ns_numpy = pred_ns[0].cpu().detach().numpy()


            # Task:
            all_a_rewards = 3 * pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 5]) - 0.5 * abs(pred_ns_numpy[:, 0])

            greedy_select = int(np.argmax(all_a_rewards))
            if greedy_select > 500:  # new random action
                select_ra_ornot.append(0)
            elif greedy_select == 0:  # previous best action (recommendation)
                select_ra_ornot.append(2)
            else:  # Noise + previous best action
                select_ra_ornot.append(1)

            choose_a = A_array_numpy[greedy_select]
            pred = pred_ns_numpy[greedy_select]

            obs_, r, done, _ = env.step(choose_a)

            gt = np.copy(obs_)
            loss = np.mean((gt - pred) ** 2)
            pred_loss.append(loss)

            A.append(choose_a)
            S.append(obs)
            NS.append(obs_)
            obs = np.copy(obs_)
            R.append(r)
            DONE.append(done)

            if done:
                print("bad case: done->1, loss: %5f and r: %5f " % (np.mean(pred_loss), r))
                obs = env.reset()
                choose_a = call_max_reward_action()
                choose_a = choose_a.cpu().detach().numpy()

    S, A, NS, R, DONE, select_ra_ornot = np.array(S), np.array(A), np.array(NS), np.array(R), np.array(DONE), np.array(
        select_ra_ornot)

    train_data_num = int(step_num * 0.8)
    idx_list = np.arange(step_num)
    np.random.shuffle(idx_list)
    train_SAS = [S[idx_list][:train_data_num],
                 A[idx_list][:train_data_num],
                 NS[idx_list][:train_data_num],
                 R[idx_list][:train_data_num],
                 DONE[idx_list][:train_data_num]]
    test_SAS = [S[idx_list][train_data_num:],
                A[idx_list][train_data_num:],
                NS[idx_list][train_data_num:],
                R[idx_list][train_data_num:],
                DONE[idx_list][train_data_num:]]

    return train_SAS, test_SAS, choose_a, select_ra_ornot, DONE


def call_max_reward_action():
    action_rewards = torch.stack((train_data.R, test_data.R))
    a_id = torch.argmax(action_rewards)
    a_choose = torch.stack((train_data.A, test_data.A))[a_id]
    return a_choose


class SAS_data(Dataset):
    def __init__(self, SAS_data):
        self.root = log_path

        self.all_S = SAS_data[0]
        self.all_A = SAS_data[1]
        self.all_NS = SAS_data[2]
        self.all_R = SAS_data[3]
        self.all_DONE = SAS_data[4]

    def __getitem__(self, idx):
        S = self.all_S[idx]
        A = self.all_A[idx]
        NS = self.all_NS[idx]
        R = np.asarray([self.all_R[idx]])
        DONE = np.asarray([self.all_DONE[idx]])

        self.S = torch.from_numpy(S.astype(np.float32)).to(device)
        self.A = torch.from_numpy(A.astype(np.float32)).to(device)
        self.NS = torch.from_numpy(NS.astype(np.float32)).to(device)
        self.R = torch.from_numpy(R.astype(np.float32)).to(device)
        self.DONE = torch.from_numpy(DONE.astype(np.float32)).to(device)

        sample = {'S': self.S, 'A': self.A, "NS": self.NS, "R": self.R, "DONE": self.DONE}

        return sample

    def __len__(self):
        return (len(self.all_S))

    def add_data(self, SAS_data):
        S_data, A_data, NS_data, all_R, all_DONE = SAS_data
        self.all_S = np.vstack((self.all_S, S_data))
        self.all_A = np.vstack((self.all_A, A_data))
        self.all_NS = np.vstack((self.all_NS, NS_data))
        self.all_R = np.hstack((self.all_R, all_R))
        self.all_DONE = np.hstack((self.all_DONE, all_DONE))
        # return [self.all_S, self.all_A, self.all_NS, self.all_R]


def train_dyna_sm(train_dataset, test_dataset):
    min_loss = + np.inf
    abort_learning = 0
    decay_lr = 0
    num_epoch = 500
    all_train_L, all_valid_L = [], []

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=False)

    for epoch in range(num_epoch):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        sm_model.train()
        for batch in train_dataloader:
            S, A, NS, R, DONE = batch["S"], batch["A"], batch["NS"], batch['R'], batch['DONE']
            pred_NS = sm_model.forward(S, A)
            loss = sm_model.loss(pred_NS, NS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        sm_model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                S, A, NS,DONE = batch["S"], batch["A"], batch["NS"], batch['DONE']
                pred_NS = sm_model.forward(S, A)
                loss = sm_model.loss(pred_NS, NS)
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            PATH = log_path + '/best_model.pt'
            torch.save(sm_model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
            decay_lr += 1
        # scheduler.step(avg_valid_L)
        np.savetxt(log_path + "training_L.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(all_valid_L))

        if abort_learning > 10:
            break
        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    print("valid_loss:", min_loss)
    return min_loss

    # plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    # plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    # plt.title("Learning Curve")
    # plt.legend()
    # plt.savefig(log_path + "lc.png")
    # plt.show()


if __name__ == "__main__":

    Train_flag = False
    mode = 5
    p.connect(p.DIRECT)

    smrl_all_dof_r_logger = []
    for dof in [12]:
        random.seed(2022)
        np.random.seed(2022)
        print("DOF:", dof)
        log_path = '../data/dof%d/smrl_model/' % dof
        os.makedirs(log_path,exist_ok=True)
        initial_para = np.loadtxt("../controller100/control_para/para.csv")
        para_dir = "../controller100/control_para/dof%d/"% dof
        dst = para_dir+"para_range.csv"
        para_space = np.loadtxt(dst)

        if mode != 5:
            env = OSM_Env(dof, initial_para, para_space,
                          urdf_path="../CAD2URDF/V000/urdf/V000.urdf",
                          data_save_pth=log_path)
        else:
            env = 0

        if mode == 0:
            NUM_EACH_CYCLE = 6
            num_cycles = 100
            batchsize = 6
            lr = 1e-4
            for sub_process in range(3):
                print('sub_process: ', sub_process)
                torch.manual_seed(sub_process)

                sm_model = FastNN(18 + 16, 18)
                sm_model.to(device)

                os.makedirs(log_path + "train/%ddata/CYCLE_%d/" % (num_cycles, NUM_EACH_CYCLE), exist_ok=True)

                os.makedirs(log_path + "train/%ddata/CYCLE_%d/%d/" % (num_cycles, NUM_EACH_CYCLE, sub_process),
                            exist_ok=True)

                log_path_ = log_path + "train/%ddata/CYCLE_%d/%d/" % (num_cycles, NUM_EACH_CYCLE, sub_process)

                optimizer = torch.optim.Adam(sm_model.parameters(), lr=lr)
                choose_a = np.random.uniform(-1, 1, size=16)
                train_SAS, test_SAS, choose_a, sele_list, DONE = collect_dyna_sm_data(step_num=NUM_EACH_CYCLE, use_policy=0,
                                                                                      choose_a=choose_a)
                train_data = SAS_data(SAS_data=train_SAS)
                test_data = SAS_data(SAS_data=test_SAS)

                log_valid_loss = []
                log_action_choose = []
                t2 = 0
                t1 = 0
                for epoch_i in range(num_cycles):

                    print(epoch_i, "time used: ", t2-t1)
                    t1 = time.time()
                    sm_train_valid_loss = train_dyna_sm(train_data, test_data)

                    train_SAS, test_SAS, choose_a, sub_sele_list, DONE = collect_dyna_sm_data(step_num=NUM_EACH_CYCLE,
                                                                                              use_policy=1,
                                                                                              choose_a=choose_a)
                    train_data.add_data(train_SAS)
                    test_data.add_data(test_SAS)

                    sele_list = np.hstack((sele_list, sub_sele_list))
                    log_valid_loss.append(sm_train_valid_loss)
                    log_action_choose.append(choose_a)
                    t2 = time.time()

                    # save data
                    if ((epoch_i + 1) % 50 == 0) or (epoch_i == 165):
                        PATH = log_path_ + "/model_%d" % (epoch_i + 1)
                        torch.save(sm_model.state_dict(), PATH)
                        np.savetxt(log_path_ + "sm_valid_loss.csv", np.asarray(log_valid_loss))
                        np.savetxt(log_path_ + "sm_action_choose.csv", np.asarray(log_action_choose))
                        np.savetxt(log_path_ + "choice_range.csv", sele_list)

        if mode == 0.5:
            NUM_EACH_CYCLE = 6
            batchsize = 6
            max_num_cycles = 300
            lr = 1e-4
            sub_process = 9
            print('sub_process: ', sub_process)
            torch.manual_seed(sub_process)

            sm_model = FastNN(18 + 16, 18)
            sm_model.to(device)

            log_path = log_path + "train/auto_stop/CYCLE_%d/%d/" % (NUM_EACH_CYCLE, sub_process)
            os.makedirs(log_path, exist_ok=True)

            optimizer = torch.optim.Adam(sm_model.parameters(), lr=lr)
            choose_a = np.random.uniform(-1, 1, size=16)
            train_SAS, test_SAS, choose_a, sele_list, done_list = collect_dyna_sm_data(step_num=NUM_EACH_CYCLE,
                                                                                       use_policy=0,
                                                                                       choose_a=choose_a)
            train_data = SAS_data(SAS_data=train_SAS)
            test_data = SAS_data(SAS_data=test_SAS)

            log_valid_loss, log_action_choose, log_rewards, log_done = [], [], [], []
            reward_bar = -np.inf
            loss_bar = np.inf
            abort_count = 0
            max_abort_count = 50
            for epoch_i in range(max_num_cycles):
                print('Epoch:%d, Abort_count:%d:' % (epoch_i, abort_count))

                sm_train_valid_loss = train_dyna_sm(train_data, test_data)
                train_SAS, test_SAS, choose_a, sub_sele_list, done_list = collect_dyna_sm_data(step_num=NUM_EACH_CYCLE,
                                                                                               use_policy=1,
                                                                                               choose_a=choose_a)
                train_data.add_data(train_SAS)
                test_data.add_data(test_SAS)
                sele_list = np.hstack((sele_list, sub_sele_list))
                log_valid_loss.append(sm_train_valid_loss)
                log_action_choose.append(choose_a)
                accumulated_R = np.sum(train_SAS[3]) + np.sum(test_SAS[3])
                log_rewards.append(accumulated_R)
                log_done.append(np.sum(done_list))

                # save data
                if ((epoch_i + 1) % 50 == 0) or (epoch_i == 165):
                    PATH = log_path + "/model_%d" % (epoch_i + 1)
                    torch.save(sm_model.state_dict(), PATH)
                    np.savetxt(log_path + "sm_valid_loss.csv", np.asarray(log_valid_loss))
                    np.savetxt(log_path + "sm_action_choose.csv", np.asarray(log_action_choose))
                    np.savetxt(log_path + "choice_range.csv", sele_list)
                    np.savetxt(log_path + "sm_rewards.csv", np.asarray(log_rewards))
                    np.savetxt(log_path + "epoch_done.csv", np.asarray(log_done))

                print("R_per_epoch: ", accumulated_R)
                print("---------------------")

                if (accumulated_R > reward_bar):
                    abort_count = 0
                    reward_bar = accumulated_R
                    loss_bar = sm_train_valid_loss
                else:
                    print('done_list: ', done_list)
                    abort_count += 1

                if abort_count >= max_abort_count:
                    break

        if mode == 1:
            NUM_EACH_CYCLE = 6
            mean_list = []
            epoch_num = 100

            for sub_prcess in [9]:
                sub_log_path = log_path + "train/%ddata/CYCLE_%d/%d/" % (epoch_num, NUM_EACH_CYCLE, sub_prcess)
                sm_valid_loss = np.loadtxt(sub_log_path + 'sm_valid_loss.csv')
                # plt.plot(X,sm_valid_loss,label='id: %d'%sub_prcess)
                mean_list.append(sm_valid_loss)

            mean_all_data = np.mean(mean_list, axis=0)
            std_all_data = np.std(mean_list, axis=0)

            X = np.asarray(range(len(mean_all_data))) * NUM_EACH_CYCLE

            plt.fill_between(X, mean_all_data - std_all_data, mean_all_data + std_all_data, alpha=0.2)
            plt.plot(X, mean_all_data)
            plt.legend()
            plt.show()

            # NUM_EACH_CYCLE = 6
            # mean_list = []


            # epoch_num = 1000
            # sub_process = 5
            # sub_log_path = log_path + "train/%ddata/CYECLE_%d/%d/" % (epoch_num, NUM_EACH_CYCLE, sub_process)
            # sm_valid_loss = np.loadtxt(sub_log_path + 'sm_valid_loss.csv')
            # mean_list.append(sm_valid_loss)
            # train_NS = np.loadtxt(sub_log_path + 'trainNS.csv')
            # test_NS = np.loadtxt(sub_log_path + 'test_NS.csv')
            # for i in range(100):
            #     plt.scatter(test_NS[:,1],test_NS[:,2])
            #
            # checkp = 44
            # plt.scatter(test_NS[checkp*2, 1], test_NS[checkp*2, 2])
            # plt.show()

        if mode == 2:
            sub_process = 9
            NUM_EACH_CYCLE = 6
            torch.manual_seed(sub_process)

            sub_log_path = log_path + "train/100data/CYCLE_%d/%d/" % (NUM_EACH_CYCLE, sub_process)
            sm_model = FastNN(18 + 16, 18)
            sm_model.to(device)

            data_num = 100
            sm_model.load_state_dict(torch.load(sub_log_path + 'model_%d' % data_num, map_location=torch.device(device)))
            # sm_model.load_state_dict(torch.load(sub_log_path + 'best_model.pt', map_location=torch.device(device)))

            log_path += '/test/'
            try:
                os.mkdir(log_path)
            except OSError:
                pass
            # env.spt = 1 / 120
            test_sm(sm_model, env, log_path, TASK='f', eval_epoch_num=10)

        # Show pred plots
        if mode == 4:
            data_num = 600 * (2 ** 6)
            robot_state = ['x', 'y', 'z', 'roll', 'pitch', 'yaw',
                           'M0', 'M1', 'M2', 'M3', 'M4',
                           'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11']

            pred = np.loadtxt(log_path + '/sm_mode/data%d/pred.csv' % data_num)
            gt = np.loadtxt(log_path + '/sm_mode/data%d/gt.csv' % data_num)

            num = 100
            print(np.mean((pred - gt) ** 2))

            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(20, 12))

            for i in range(18):
                ax_x = i // 6
                ax_y = i % 6

                axs[ax_x, ax_y].plot(pred[:num, i], label='pred')
                axs[ax_x, ax_y].plot(gt[:num, i], label='gt')
                loss_value = np.mean((pred[:num, i] - gt[:num, i]) ** 2)
                content = '%s, loss: %f' % (robot_state[i], float(loss_value))
                axs[ax_x, ax_y].set_title(content)
            plt.legend()
            plt.show()

        if mode == 5:
            NUM_EACH_CYCLE = 6
            data_num1 = 100
            data_logger = []
            for sub_process in range(3):
                print("sub_process", sub_process)
                torch.manual_seed(sub_process)

                sm_log_path = log_path + "/sm_model/%d/" % (sub_process)
                sm_model = FastNN(18 + 16, 18)

                smrl_model_path = log_path + "rl_model(sm)/%d/" % sub_process

                os.makedirs(smrl_model_path, exist_ok=True)

                # Load self-model
                sm_model.load_state_dict(
                    torch.load(sm_log_path + 'model_100', map_location=torch.device(device)))
                sm_model.to(device)
                sm_model.eval()

                env = OSM_Env(dof, initial_para, para_space, urdf_path="../CAD2URDF/V000/urdf/V000.urdf",
                              data_save_pth=None, sm_world=sm_model)

                if Train_flag:
                    model = PPO("MlpPolicy", env, n_steps=6, verbose=0, batch_size=6)
                    train_agent_with_sm(env, model, smrl_model_path)
                else:
                    model = PPO.load(smrl_model_path + "best_model", env)
                    r, y = evaluate_RL(env, model)
                    data_logger.append(r)

            smrl_all_dof_r_logger.append(data_logger)
            np.savetxt("../paper_data/smrl_logger.csv", np.asarray(smrl_all_dof_r_logger))






