import pybullet as p
import time
import pybullet_data as pd
import gym
import random
from controller100.control import *
import numpy as np
import scipy.linalg as linalg



class OSM_Env(gym.Env):
    def __init__(self, dof, render,para,noise_para, data_save_pth, robot_camera=False, urdf_path='CAD2URDF'):
        self.save_pth = data_save_pth
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.robotid = None
        self.v_p = None
        self.q = None
        self.p = None
        self.last_q = None
        self.last_p = None
        self.last_last_obs = [0]*18
        self.last_obs = [0]*18
        self.obs = [0]*18
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1. / 240  # decrease the value if it is too slow.
        self.force = 1.8
        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.sub_step_num = 16
        self.dof = dof
        self.robot_type = [[3,6],
                           [0,3,6,9],
                           [0,3,4,6,7,9,],
                           [0,1,3,4,6,7,9,10],
                           [0,1,3,4,5,6,7,8,9,10],
                           [0,1,2,3,4,5,6,7,8,9,10,11]]
        self.robot_actuated = self.robot_type[(self.dof-1)//2 ]
        self.noise_para = noise_para
        self.para = para
        self.initial_values = sin_move(0, para, sep=self.sub_step_num)

        obs = self.reset()
        self.action_space = gym.spaces.Box(low=-np.ones(16), high=np.ones(16))
        self.observation_space = gym.spaces.Box(low=-np.ones_like(obs)*np.inf, high=np.ones_like(obs)*np.inf)

        self.log_obs = []
        self.log_action = []
        self.count = 0

        p.setAdditionalSearchPath(pd.getDataPath())

    def get_obs(self):
        self.last_last_obs = np.copy(self.last_obs)
        self.last_obs = np.copy(self.obs)
        self.last_p = self.p
        self.last_q = self.q
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        # Delta position and orientation
        self.v_p = self.p - self.last_p
        self.v_q = self.q - self.last_q


        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()
        self.obs = np.concatenate([self.v_p, self.q, jointVals])
        obs_data = np.concatenate([self.last_last_obs, self.last_obs, self.obs])

        return obs_data

    def act(self, sin_para):
        sin_para = sin_para*self.noise_para + self.para
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
                time.sleep(self.sleep_time)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)
        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        return self.get_obs()

    def step(self, a):

        self.act(a)
        obs = self.get_obs()

        pos, _ = self.robot_location()
        r = 2 * obs[1] - np.abs(obs[0])

        done = self.check()

        self.log_obs.append(obs)
        self.log_action.append(a)
        
        if self.count%100 ==0:
            np.savetxt(self.save_pth + "/log_obs.csv", np.asarray(self.log_obs))
            np.savetxt(self.save_pth + "/log_action.csv", np.asarray(self.log_action))

        self.count +=1


        return obs, r, done, {}

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
        # elif abs(pos[2]) < 0.22:
        #     abort_flag = True
        else:
            abort_flag = False
        return abort_flag


if __name__ == '__main__':
    env = OSM_Env(robot_camera=False,render = True,dof = 12, urdf_path="CAD2URDF/V000/urdf/V000.urdf")
    env.sleep_time = 1/240

    # Run Sin Gait Baseline
    data_path = "controller100/control_para/dof12/"
    para = np.loadtxt(data_path + "0.csv")

    # Run the trajectory commands
    # best_action_file = np.loadtxt("test.csv")
    results = []

    for epoch in range(100):
        print(epoch)
        fail = 0
        result = 0
        env.reset()
        action_logger = []
        for i in range(6):
            time1 = time.time()
            action = np.random.normal(para,scale = [0.4,0.4,
                                                    0.5,0.5,0.5,0.5,
                                                    0.4,0.4,
                                                    0.4,0.4,
                                                    0.1,0.1,0.1,0.1,0.1,0.1
                                                    ])
            action_logger.append(action)
            obs, r, done, _ = env.step(action)
            result = env.robot_location()
            print(result, r, done)
            time2 = time.time()
            print(0.12 - (time2 - time1))

        results.append(result[0])
        # np.savetxt("sin_gait_action.csv",action_logger)
    # np.savetxt("perfect_self_model/analysis/results.csv",np.asarray(results))
