import time
import pybullet_data as pd
import gym
import numpy as np
import scipy.linalg as linalg
import random
import cv2
import math as m
import pybullet as p

def Rz_2D(theta):
    return np.array([[m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta), 0]])


def World2Local(pos_base, ori_base, pos_new):
    psi, theta, phi = ori_base
    R = Rz_2D(phi)
    R_in = R.T
    pos = np.asarray([pos_base[:2]]).T
    R = np.hstack((R_in, np.dot(-R_in, pos)))
    pos2 = list(pos_new[:2]) + [1]
    vec = np.asarray(pos2).T
    local_coo = np.dot(R, vec)
    return local_coo


class OpticalEnv(gym.Env):
    def __init__(self, name, robot_camera=False, camera_capture=False, data_cll_flag=False, urdf_path='CAD2URDF',
                 ground_type="rug", rand_fiction=False, rand_torque=False, rand_pos=False,CONSTRAIN = False):

        self.name = name
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1. / 240

        self.n_sim_steps = 60  # 1 step = 0.00387 s -> 0.2322 s/step
        self.inner_motor_index = [0, 3, 6, 9]
        self.middle_motor_index = [1, 4, 7, 10]
        self.outer_motor_index = [2, 5, 8, 11]
        self.CONSTRAIN = CONSTRAIN
        self.original_motor_action_space = np.pi/3
        self.motor_action_space = np.asarray([np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10])

        self.motor_action_space_shift = np.asarray([0, -1.2, 1,
                                                    0.5, -1.2, 1,
                                                    -0.5, -1.2, 1,
                                                    0, -1.2, 1])
        self.log_state = [0, 0, 0, 0, 0, 0]
        self.urdf_path = urdf_path
        self.camera_capture = camera_capture
        self.counting = 0
        self.robot_camera = robot_camera
        if rand_fiction == True:
            self.friction = random.uniform(0.5, 0.99)
            # print("F:",self.friction)
        else:
            self.friction = 0.99
        self.pos_rand_flag = rand_pos
        self.force_rand_flag = rand_torque
        self.force = 1.8
        self.robot_view_path = None
        self.data_collection = data_cll_flag
        self.ground_type = ground_type
        self.internal_f_state = []
        self.ov_input = []

        self.reset()

        p.setAdditionalSearchPath(pd.getDataPath())


    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)

        self.terrian_init()


    def step(self, a):
        obs, r, done = 0,0,0
        return obs, r, done, {}


    def terrian_init(self):

        random.seed(5)
        heightPerturbationRange = 0.05
        numHeightfieldRows = 100
        numHeightfieldColumns = 100
        heightfieldData = np.zeros(shape=[numHeightfieldColumns, numHeightfieldRows], dtype=np.float)
        for i in range(int(numHeightfieldColumns / 2)):
            for j in range(int(numHeightfieldRows)):
                n1 = 0
                n2 = 0
                if j > 0:
                    n1 = heightfieldData[i, j - 1]
                if i > 0:
                    n2 = heightfieldData[i - 1, j]
                else:
                    n2 = n1
                noise = random.uniform(-heightPerturbationRange,
                                       heightPerturbationRange)
                heightfieldData[i, j] = (n1 + n2) / 2 + noise

        heightfieldData_inv = heightfieldData[::-1, :]
        heightfieldData_2 = np.concatenate((heightfieldData_inv, heightfieldData))
        # print(heightfieldData_2)

        col, row = heightfieldData_2.shape
        heightfieldData_2 = heightfieldData_2.reshape(-1)

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, heightfieldData=heightfieldData_2,
                                              meshScale=[0.05, 0.05, 0.1],
                                              numHeightfieldRows=row, numHeightfieldColumns=col)
        terrain = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])


if __name__ == '__main__':
    robot_idx = 0
    print("robot_idx:", robot_idx)
    name = 'V0%02d' % robot_idx
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    RAND_FIRCTION = False
    RAND_T = False
    RAND_P = False
    env = OpticalEnv(name,
                     robot_camera=False,
                     urdf_path="CADandURDF/robot_repo/V000_cam/urdf/V000_cam.urdf",
                     rand_fiction=RAND_FIRCTION,
                     rand_torque=RAND_T,
                     rand_pos= RAND_P,
                     CONSTRAIN=True
                     )


    for i in range(10000):
        print(1)
        p.stepSimulation()
        time.sleep(1/240)
