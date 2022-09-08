import pybullet as p
import time
import pybullet_data as pd
import gym
import random
from controller100.control import *
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


class OSM_Env(gym.Env):
    def __init__(self, name, robot_camera = False, urdf_path = 'CAD2URDF'):

        self.name = name
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5 # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1./240 # decrease the value if it is too slow.
        self.force = 1.8
        self.n_sim_steps = 30
        self.motor_action_space = np.pi/3
        self.log_state = [0,0,0,0,0,0]
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.clock = 0
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None


        obs = self.reset()
        self.action_space = gym.spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof))
        self.observation_space = gym.spaces.Box(low=-np.ones_like(obs)*np.inf, high=np.ones_like(obs)*np.inf)

        p.setAdditionalSearchPath(pd.getDataPath())

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
            p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=a[i], force = self.force, maxVelocity =self.maxVelocity)

        for i in range(self.n_sim_steps):
            p.stepSimulation()
            self.clock +=1
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
        # textureId = p.loadTexture("grass.png")
        # wall_textureId = p.loadTexture("wall_picture.png")
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        # try:
        #     p.removeBody(self.robotid)
        # except:
        #     print('start')

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation,flags=p.URDF_USE_SELF_COLLISION, useFixedBase=0)
        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.i = 0
        self.dof = p.getNumJoints(self.robotid)
        self.clock = 0


        return self.get_obs()

    def step(self, a):
        a = np.clip(a, -1, 1)
        a *= self.motor_action_space
        self.act(a)
        obs = self.get_obs()
        pos, _ = self.robot_location()
        r = 100*obs[1] - 50 * np.abs(obs[0])
        done = self.check()
        return obs, r, done, {}


    def robot_location(self):
    #     if call this, the absolute location of robot body will back
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        if abs(pos[0])> 0.5:
            abort_flag = True
        elif abs(ori[0]) > np.pi/6 or abs(ori[1]) > np.pi/6 or abs(ori[2]) > np.pi/6:
            abort_flag = True
        elif abs(pos[2])<0.22:
            abort_flag = True
        else:
            abort_flag = False
        return abort_flag



if __name__ == '__main__':
    robot_idx = 0
    print("robot_idx:", robot_idx)
    name = 'V%03d'%robot_idx
    physicsClient = p.connect(p.GUI)

    env = OSM_Env(name,robot_camera = False,urdf_path = "CAD2URDF/%s/%s/urdf/%s.urdf"%(name[:3],name,name))
    env.sleep_time = 1/240

    # Run Sin Gait Baseline
    data_path = "controller100/control_para/logV000/"
    para = np.loadtxt(data_path + "0.csv")

    # Run the trajectory commands
    # best_action_file = np.loadtxt("test.csv")
    results = []

    for epoch in range(1):
        print(epoch)
        fail = 0
        result = 0
        env.reset()
        action_logger = []
        for i in range(100):
            time1 = time.time()
            action = sin_move(i,para)
            action_logger.append(action)
            obs, r, done, _ = env.step(action)
            result = env.robot_location()
            print(result,r, done)
            time2 = time.time()
            print(0.12 - (time2-time1))

        results.append(result[0])
        # np.savetxt("sin_gait_action.csv",action_logger)
    # np.savetxt("perfect_self_model/analysis/results.csv",np.asarray(results))
