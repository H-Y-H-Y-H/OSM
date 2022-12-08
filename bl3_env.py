import pybullet as p
import time
import pybullet_data as pd
import gym
from controller100.control import *
import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class OSM_Env(gym.Env):
    def __init__(self, dof, para, noise_para, data_save_pth, robot_camera=False, urdf_path='CAD2URDF',
                 sm_world=None, TASK='f'):
        self.choose_a = None
        self.save_pth = data_save_pth  # save SAS data


        self.spt = 0  # decrease the value if it is too slow.

        self.robotid = None
        self.v_p = None
        self.q = None
        self.p = None
        self.last_q = None
        self.last_p = None
        self.last_last_obs = [0] * 18
        self.last_obs = [0] * 18
        self.obs = [0] * 18

        self.noise_para = noise_para
        self.para = para
        self.sub_step_num = 16
        self.initial_values = sin_move(0, para, sep=self.sub_step_num)

        self.a = self.para
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s

        self.force = 1.8
        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.dof = dof
        self.sm_pred_error_list = []

        self.robot_type = [[3, 6],
                           [0, 3, 6, 9],
                           [0, 3, 4, 6, 7, 9],
                           [0, 1, 3, 4, 6, 7, 9, 10],
                           [0, 1, 3, 4, 5, 6, 7, 8, 9, 10],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

                    # dof 4, C0: [0, 3, 6, 9],
        if self.dof <=12:
            self.robot_actuated = self.robot_type[(self.dof - 1) // 2]
        elif self.dof == 200:
            self.robot_actuated = [0, 9]
        elif self.dof == 201:
            self.robot_actuated = [4, 7]
        elif self.dof == 202:
            self.robot_actuated = [1, 10]
        elif self.dof == 203:
            self.robot_actuated = [5, 8]
        elif self.dof == 204:
            self.robot_actuated = [2, 11]
        elif self.dof == 400:
            self.robot_actuated = [0, 3, 6, 9]
        elif self.dof == 401:
            self.robot_actuated = [0, 3, 7, 10]
        elif self.dof == 402:
            self.robot_actuated = [0, 3, 8, 11]
        elif self.dof == 403:
            self.robot_actuated = [1, 4, 6, 9]
        elif self.dof == 404:
            self.robot_actuated = [1, 4, 7, 10]
        elif self.dof == 405:
            self.robot_actuated = [1, 4, 8, 11]
        elif self.dof == 406:
            self.robot_actuated = [2, 5, 6, 9]
        elif self.dof == 407:
            self.robot_actuated = [2, 5, 7, 10]
        elif self.dof == 408:
            self.robot_actuated = [2, 5, 8, 11]
        elif self.dof == 600:
            self.robot_actuated = [0, 1, 2, 9, 10, 11]
        elif self.dof == 601:
            self.robot_actuated = [0, 1, 3, 6, 9, 10]
        elif self.dof == 602:
            self.robot_actuated = [0, 1, 3, 6, 9, 10]
        elif self.dof == 603:
            self.robot_actuated = [0, 1, 4, 6, 9, 10]
        elif self.dof == 604:
            self.robot_actuated = [0, 1, 5, 8, 9, 10]
        elif self.dof == 605:
            self.robot_actuated = [0, 2, 3, 6, 9, 11]
        elif self.dof == 606:
            self.robot_actuated = [0, 2, 4, 7, 9, 11]
        elif self.dof == 607:
            self.robot_actuated = [0, 2, 5, 8, 9, 11]
        elif self.dof == 608:
            self.robot_actuated = [1, 2, 3, 6, 10, 11]
        elif self.dof == 609:
            self.robot_actuated = [0, 3, 5, 6, 8, 9]
        elif self.dof == 610:
            self.robot_actuated = [0, 4, 5, 7, 8, 9]
        elif self.dof == 611:
            self.robot_actuated = [3, 4, 5, 6, 7, 8]
        elif self.dof == 612:
            self.robot_actuated = [1, 3, 4, 6, 7, 10]
        elif self.dof == 613:
            self.robot_actuated = [2, 3, 4, 6, 7, 11]
        elif self.dof == 614:
            self.robot_actuated = [1, 3, 5, 6, 8, 10]
        elif self.dof == 615:
            self.robot_actuated = [2, 3, 5, 6, 8, 11]
        elif self.dof == 800:
            self.robot_actuated = [0, 2, 3, 5, 6, 8, 9, 11]
        elif self.dof == 801:
            self.robot_actuated = [0, 1, 2, 3, 6, 9, 10, 11]
        elif self.dof == 802:
            self.robot_actuated = [0, 1, 2, 4, 7, 9, 10, 11]
        elif self.dof == 803:
            self.robot_actuated = [0, 1, 2, 5, 8, 9, 10, 11]
        elif self.dof == 804:
            self.robot_actuated = [0, 3, 4, 5, 6, 7, 8, 9]
        elif self.dof == 805:
            self.robot_actuated = [1, 3, 4, 5, 6, 7, 8, 10]
        elif self.dof == 806:
            self.robot_actuated = [2, 3, 4, 5, 6, 7, 8, 11]
        elif self.dof == 1000:
            self.robot_actuated = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
        elif self.dof == 1001:
            self.robot_actuated = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        elif self.dof == 1002:
            self.robot_actuated = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]
        elif self.dof == 1003:
            self.robot_actuated = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]
        elif self.dof == 1004:
            self.robot_actuated = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11]
            #####  11            2
            ##       10       1
            ######      9   0
            #             @
            ###         6   3
            ###       7        4
            ##      8            5



        self.sm_model = sm_world
        self.sm_model_world = False

        self.action_space = gym.spaces.Box(low=-np.ones(16), high=np.ones(16))
        self.observation_space = gym.spaces.Box(low=-np.ones(18) * np.inf, high=np.ones(18) * np.inf)
        self.log_obs = []
        self.log_action = []
        self.count = 0
        self.task = TASK

        p.setAdditionalSearchPath(pd.getDataPath())

    def get_obs(self):
        self.last_p = self.p
        self.last_q = self.q
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        # Delta position and orientation
        self.v_p = self.p - self.last_p
        self.v_q = self.q - self.last_q
        if self.v_q[2] > 1.57:
            self.v_q[2] = self.q[2] - self.last_q[2] - 2 * np.pi
        elif self.v_q[2] < -1.57:
            self.v_q[2] = (2 * np.pi + self.q[2]) - self.last_q[2]

        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()

        if not self.sm_model_world:
            self.obs = np.concatenate([self.v_p, self.q, jointVals])

        # print(np.mean((self.obs - new_obs)**2))

        else:
            # Delta position and orientation
            real_obs = np.concatenate([self.v_p, self.q, jointVals])

            obs_data = self.obs
            obs_data = torch.from_numpy(obs_data).to(device, dtype=torch.float)
            tensor_a = torch.from_numpy(self.a).to(device, dtype=torch.float)
            new_obs = self.sm_model.forward(obs_data, tensor_a)

            new_obs = new_obs[0].cpu().detach().numpy()

            self.sm_pred_error_list.append(np.mean((real_obs - new_obs) ** 2))
            # new_obs = np.copy(real_obs)

            self.last_last_obs = np.copy(self.last_obs)
            self.last_obs = np.copy(self.obs)
            self.obs = np.copy(new_obs)

        return self.obs

    def act(self, a):
        self.a = a
        sin_para = a * self.noise_para + self.para
        for sub_step in range(self.sub_step_num):
            a = sin_move(sub_step, sin_para, sep=self.sub_step_num)
            a = np.clip(a, -1, 1)
            a *= self.motor_action_space

            for i in range(12):
                if i in self.robot_actuated:
                    pos_value = a[i]
                else:
                    pos_value = self.initial_values[i]

                p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos_value,
                                        force=self.force,
                                        maxVelocity=self.maxVelocity)

            for i in range(self.n_sim_steps):
                p.stepSimulation()

                if self.render:
                    # Capture Camera
                    if self.camera_capture == True:
                        basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
                        basePos_list = [basePos[0], basePos[1], 0.3]
                        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                                     cameraTargetPosition=basePos_list)  # fix camera onto model
            time.sleep(self.spt)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.22]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)
        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

        # Move to initial state:
        for i in range(12):
            pos_value = self.initial_values[i]
            p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos_value,
                                    force=self.force,
                                    maxVelocity=100)
        for _ in range(40):
            p.stepSimulation()
        # time.sleep(1/240)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()
        self.obs = np.concatenate([self.p, self.q, jointVals])

        return self.get_obs()

    def step(self, a):

        # if not self.sm_model_world:
        #     S_array = np.asarray([self.obs] * 2560)
        #     A_array_numpy = np.random.uniform(-1, 1, size=(2560, 16))
        #
        #     S_array = torch.from_numpy(S_array.astype(np.float32)).to(device)
        #     A_array = torch.from_numpy(A_array_numpy.astype(np.float32)).to(device)
        #     pred_ns = self.sm_model.forward(S_array, A_array)
        #     pred_ns_numpy = pred_ns[0].cpu().detach().numpy()
        #     if self.task == "f":
        #         all_a_rewards = 3 * pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 5]) - 0.5 * abs(pred_ns_numpy[:, 0])
        #     else:
        #         all_a_rewards = np.zeros(10)
        #
        #     greedy_select = int(np.argmax(all_a_rewards))
        #     self.choose_a = A_array_numpy[greedy_select]

        self.act(a)

        obs = self.get_obs()

        pos, _ = self.robot_location()

        r = 3 * obs[1] - abs(obs[5]) - 0.5 * abs(obs[0]) + 1

        # r -= np.mean((self.choose_a - a) ** 2) * 0.1

        Done = self.check()
        if Done:
            r -=2

        self.log_obs.append(obs)
        self.log_action.append(a)

        if (self.count % 100 == 0) and (self.save_pth != None):
            np.savetxt(self.save_pth + "/log_obs.csv", np.asarray(self.log_obs))
            np.savetxt(self.save_pth + "/log_action.csv", np.asarray(self.log_action))
            np.save(self.save_pth + "/log_pred_error_every_epoch.npy", np.asarray(self.sm_pred_error_list))
        self.count += 1

        return obs, r, Done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     abort_flag = True
        if abs(ori[0]) > np.pi / 2 or abs(ori[1]) > np.pi / 2:
            abort_flag = True
        elif pos[1] < -0.04:
            abort_flag = True
        elif abs(pos[0]) > 0.2:
            abort_flag = True
        else:
            abort_flag = False
        return abort_flag


if __name__ == '__main__':

    # Run Sin Gait Baseline
    data_path = "controller100/control_para/dof12/"
    log_path = 'data/origin_para/dof12/'
    inital_para = np.loadtxt("controller100/control_para/dof12/0.csv")
    para_space = np.asarray([0.6, 0.6,
                             0.6, 0.6, 0.6, 0.6,
                             0.5, 0.5,
                             0.6, 0.6,
                             0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # para = np.loadtxt(data_path + "0.csv")

    # Run the trajectory commands
    action_file = np.loadtxt("data/origin_para/dof12/sm_model/train/1000data/CYECLE_6/5/trainA.csv")
    results = []
    render_flag = True
    env = OSM_Env(12, render_flag, inital_para, para_space, urdf_path="CAD2URDF/V000/urdf/V000.urdf")
    env.spt = 1 / 240

    for epoch in range(100):
        print(epoch)
        fail = 0
        result = 0
        env.reset()
        action_logger = []
        for i in range(43, 44):
            for j in range(3):
                # POLICY 1: Random
                # action = []
                # for j in range(16):
                #     action.append(random.uniform(-1,1))
                # POLICY 2: Read outside data
                action = action_file[i * 3 + j]
                # action_logger.append(action)
                obs, r, done, _ = env.step(action)
                result = env.robot_location()
                # time.sleep(3)
                print(result, r, done)

        results.append(result[0])
        # np.savetxt("sin_gait_action.csv",action_logger)
    # np.savetxt("perfect_self_model/analysis/results.csv",np.asarray(results))
