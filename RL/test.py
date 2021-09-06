#!/usr/bin/env python
import numpy as np

robot_yaw1 = np.random.uniform(-1.0, 1.0, 1)
robot_yaw2 = np.random.uniform(-1.0, 1.0, 1)
robot_yaw23 = np.hstack((robot_yaw1, robot_yaw2))
print(robot_yaw23)

