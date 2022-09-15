import pybullet as p
import time
import pybullet_data as pd

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0,0,-10)
p.loadURDF("plane.urdf")
cubeStartPos = [0,0,10.5]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("cube.urdf",cubeStartPos, cubeStartOrientation)


for i in range(1000):

    p.stepSimulation()
    # time.sleep(1/240)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    if cubePos[2]>=0.5:
        print(i, cubePos[2])

# steps = 365
# time used for 10 m:
# 1/2 * 10 * t*t = 10
# t = sqrt(2) s
#sqrt(2) s = 365 step
# 1 step = 0.00387 s


p.disconnect()