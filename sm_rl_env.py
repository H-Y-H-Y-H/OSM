import pybullet as p
import time
import pybullet_data
import gym
import numpy as np
import scipy.linalg as linalg
import sys
import torch
import os
from model import *
from controller100.control import *

sys.path.append("/")
sys.path.append("../urdf_and_controllers_100_spider_robots")


if torch.cuda.is_available():
    device = 'cuda'
else: device = 'cpu'


class SM_RL_Env(gym.Env):
    def __init__(self, name,sm_model,para_data,noise, MaxVelocity=3.236, render=False, urdf_path = 'CAD2URDF',follow_camera=True, num_traj = 50, num_pred_step = 4,step_return_info = 0):
        if render:
            physicsClient = p.connect(p.GUI)
            self.render = True
        else:
            self.render = False
            physicsClient = p.connect(p.DIRECT)
        self.name = name
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = MaxVelocity # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1./240 # decrease the value if it is too slow.
        self.n_sim_steps = 30
        self.inner_motor_index =  [0,3,6, 9]
        self.middle_motor_index=  [1,4,7,10]
        self.outer_motor_index =  [2,5,8,11]
        self.motor_action_space = np.pi/3
        self.log_state = [0,0,0,0,0,0]
        self.urdf_path = urdf_path
        self.step_return_info = step_return_info
        self.last_act = np.zeros(12)
        self.follow_camera = follow_camera
        self.real_r_acc = 0

        obs = self.reset()
        self.action_space = gym.spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof))
        self.observation_space = gym.spaces.Box(low=-np.ones_like(obs)*np.inf, high=np.ones_like(obs)*np.inf)

        self.sm_model = sm_model
        self.num_pred_step = num_pred_step
        self.num_traj = num_traj
        self.last_state      = torch.zeros([1, 18],dtype=torch.float32).to(device)
        self.last_last_state = torch.zeros([1, 18],dtype=torch.float32).to(device)
        self.clock = 0
        self.para_data = para_data
        self.noise = noise





    def _render(self, mode='human', close=False):
        pass

    def coordinate_transform(self,input_state):
        # input_state[:3] -= self.log_state[:3]
        # print("transformation",input_state)
        radian = - self.log_state[5]
        # print("theta",theta)
        matrix_r = linalg.expm(
            np.cross(np.eye(3), [0, 0, 1] / linalg.norm([0, 0, 1]) * radian))  # [0,0,1] rotate z axis
        pos = np.asarray([input_state[0], input_state[1], input_state[2]])
        output_state = np.dot(matrix_r, pos)
        output_state = list(output_state[:3]) + list(input_state[3:5]) + [input_state[5] - self.log_state[5]]
        return output_state

    def get_obs(self):

        self.last_p = self.p
        self.last_q = self.q

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        self.v_p = self.p - self.last_p
        self.v_q = self.q - self.last_q

        jointInfo = [p.getJointState(self.robotid, i) for i in range(self.dof)]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()

        # body_pos:3 + body_orientation:3 + joints: 12
        # normalization [-1,1];  mean:0 std 1
        obs = np.concatenate([self.v_p, self.q, jointVals])

        return obs

    def act(self, a):

        for i in range(len(a)):
            p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=a[i], force = 1.8, maxVelocity =self.maxVelocity)
            # p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=a[i], force = 0.5, maxVelocity =self.maxVelocity)

        for i in range(self.n_sim_steps): # 30 * 1/240s
            p.stepSimulation()

            if self.follow_camera ==True:
                basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
                basePos_list = [basePos[0],basePos[1],0.3]
                p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                             cameraTargetPosition=basePos_list)  # fix camera onto model

            if self.render:
                time.sleep(self.sleep_time)

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        planeId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotid = p.loadURDF(self.urdf_path +"/%s/%s/urdf/%s.urdf"%(self.name[:3],self.name,self.name), robotStartPos, robotStartOrientation,flags=p.URDF_USE_SELF_COLLISION, useFixedBase=0)

        p.changeDynamics(self.robotid, 2, lateralFriction=0.99)
        p.changeDynamics(self.robotid, 5, lateralFriction=0.99)
        p.changeDynamics(self.robotid, 8, lateralFriction=0.99)
        p.changeDynamics(self.robotid, 11, lateralFriction=0.99)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.i = 0
        self.dof = p.getNumJoints(self.robotid)
        self.clock = 0
        self.last_state = torch.zeros([1, 18], dtype=torch.float32).to(device)
        self.last_last_state = torch.zeros([1, 18], dtype=torch.float32).to(device)
        self.real_r_acc =0
        return self.get_obs()

    def step(self, a):
        # cost_engry = np.sum(np.abs(self.last_act - a))**2
        # self.last_act = np.copy(a)
        a *= self.motor_action_space
        self.act(a)
        obs = self.get_obs()
        done = False
        all_action = []
        traj_reward_list = []
        estimated_r = 0
        for traj in range(self.num_traj):
            first_action = []
            s_ll_sub = self.last_last_state
            s_l_sub = self.last_state
            s_sub = torch.from_numpy(np.asarray([obs]).astype(np.float32)).to(device)
            traj_reward = 0
            for seq in range(self.num_pred_step):

                action = sin_move(self.clock+seq, self.para_data)
                if traj != 0:
                    action = np.random.normal(loc=action, scale=self.noise, size=None)
                    action = np.clip(action, -1, 1)
                # action = a
                # action = np.random.normal(loc=action, scale=self.noise, size=None)
                # action = np.clip(action, -1, 1)
                action = np.array([action]).astype(np.float32)
                # save the first action, which is the actions that will be selected to be executed by reward rank.
                if seq == 0:
                    first_action = action

                action = torch.from_numpy(action).to(device)
                s_c = torch.cat([s_ll_sub, s_l_sub, s_sub], -1)

                pred_ns = self.sm_model(s_c, action)

                s_ll_sub = s_l_sub
                s_l_sub = s_sub
                s_sub = pred_ns[0]
                pred_ns_numpy = pred_ns[0][0].cpu().detach().numpy()

                traj_reward += (100 * pred_ns_numpy[1] - 50 * np.abs(pred_ns_numpy[0]))
                # traj_reward += (-100 * pred_ns_numpy[1] - 50 * np.abs(pred_ns_numpy[0]))

            all_action.append(first_action[0])
            traj_reward_list.append(traj_reward/ self.num_pred_step)

            estimated_r += traj_reward

        action_idx = int(np.argmax(traj_reward_list))
        estimated_r = np.sum(traj_reward_list) / self.num_traj
        choose_next_action = all_action[action_idx]


        self.last_last_state = self.last_state
        self.last_state = torch.from_numpy(np.asarray([obs]).astype(np.float32)).to(device)


        real_r = 100 * obs[1] - 50 * np.abs(obs[0])
        hybrid_r  = (real_r*0.8 + estimated_r*0.2)


        # print(cost_engry)
        # Check:
        done = self.check()

        # print("robot_loc",self.robot_location()[0])
        self.clock +=1
        # print(self.clock)
        # self.real_r_acc += real_r
        # if self.clock == 100:
        #     print(self.robot_location())
        #     print(self.real_r_acc)

        if self.step_return_info == 0:
            return obs, choose_next_action, done, {}

        elif self.step_return_info == 1:
            return obs, hybrid_r, done, {}

        elif self.step_return_info == 2:
            return obs, real_r, done, {}

        else:
            return obs, real_r, done, {}



    def robot_location(self):
    #     if call this, the absolute location of robot body will back
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()

        if abs(pos[0])> 0.5:
            flag = True
        elif abs(ori[0]) > np.pi/6 or abs(ori[1]) > np.pi/6 or abs(ori[2]) > np.pi/6:
            flag = True
        else:
            flag = False
        return flag

if __name__ == '__main__':

    # RUN Self-model with MPC
    robot_idx = 0
    name = 'V0%02d'%robot_idx

    model_name = "data_200k_noise0.2"
    noise = 0.2
    render_switch = True
    slp_time = 0
    sm_model = FastNN(18,12)
    log_path = '../urdf_and_controllers_100_spider_robots/data_for_self_model/%s/%s/colab/'%(model_name,name)
    sm_model.load_state_dict(torch.load(log_path + "/best_model.pt", map_location=torch.device(device)))
    sm_model.eval()
    sm_model.to(device)

    sin_gait_para = np.array(np.loadtxt("controller100/control_para/log%s/0.csv" % (name)))

    env = SM_RL_Env(name,
                    sm_model=sm_model,
                    para_data= sin_gait_para,
                    render=render_switch,
                    noise = noise,
                    urdf_path="../urdf_and_controllers_100_spider_robots/CAD2URDF",
                    num_pred_step=4,
                    num_traj=50,
                    follow_camera= False
                    )

    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=120, cameraPitch=-30,
                                 cameraTargetPosition=[0,3,0.2])  # fix camera onto model
    env.sleep_time = slp_time
    # Run the trajectory commands
    # best_action_file = np.loadtxt("/Users/yuhang/Google 云端硬盘/rcfg_robot/urdf_and_controllers_100_spider_robots/data_for_self-model/data_200k_multi_input/V000/colab_linear_best/log_action1.csv")
    # best_action_file = np.loadtxt("/Users/yuhang/Google 云端硬盘/rcfg_robot/rl_baselines/SM_SAC1.csv")
    # best_action_file = np.loadtxt("test.csv")
    results = []

    SELF_MODEL_MODE =  0
    for epoch in range(3):
        print(epoch)

        fail = 0
        result = 0
        env.reset()
        action_logger = []
        action = sin_move(0, sin_gait_para)
        next_action = action
        action_list = []
        for i in range(100):
            if SELF_MODEL_MODE ==0:
                env.num_traj = 1
                action = sin_move(i, sin_gait_para)
                action = np.random.normal(loc=action, scale=noise, size=None)
                action = np.clip(action, -1, 1)
            else:
                action = next_action

            obs, next_action, done, _ = env.step(action)
            action_list.append(action)
            # if done == True:
            #     break
        np.savetxt("action_0.2noise.csv",np.asarray(action_list))
        print(env.robot_location()[0])

