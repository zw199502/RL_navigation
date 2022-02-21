import rospy
import threading
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

from gazebo_msgs.srv import SetModelState, SetModelStateRequest, GetModelState, GetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.msg import ModelState

import sys
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist,PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Empty


#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, atan2, sqrt, fabs, hypot

# build goal set, where the location of each goal needs to be 0.5m far from obstacles, 
# so as to enable all goals accessible
goal_set = [[ 0.0,  0.0], [ 0.5,  0.0], [ 1.5,  0.0], [ 2.5,  0.0], [ 4.2,  0.0], 
            [ 0.5,  1.0], [ 2.0,  1.0], [ 2.7,  1.0], [ 4.2,  1.0], [ 0.0,  2.0],
            [ 1.0,  1.7], [ 2.0,  2.0], [ 3.0,  2.5], [ 4.0,  2.3], [ 4.2,  3.3],
            [ 4.1,  4.1], [ 3.1,  4.2], [ 2.0,  3.5], [ 1.6,  4.3], [ 0.2,  3.4],
            [-0.5,  0.5], [-0.5,  1.5], [-1.5,  0.5], [-2.2,  0.3], [-3.6,  0.0],
            [-4.2,  0.3], [-4.0,  1.0], [-1.7,  1.0], [-1.0,  1.8], [-0.5,  2.4],
            [-0.9,  4.2], [-2.5,  3.0], [-3.8,  2.5], [-4.2,  4.0], [-3.0,  4.2],
            [-1.0, -1.0], [-2.0, -1.0], [-3.0, -1.7], [-4.2, -0.7], [-4.2, -1.7],
            [-3.0, -2.0], [-4.1, -3.7], [-3.5, -4.2], [-2.5, -3.8], [-1.4, -4.2],
            [-0.7, -3.5], [-0.9, -3.5], [-0.5, -1.5], [ 0.6, -0.7], [ 0.7, -1.8],
            [ 1.6, -1.6], [ 0.5, -2.5], [ 0.0, -4.0], [ 1.0, -4.0], [ 2.0, -3.8],
            [ 3.2, -3.8], [ 4.0, -4.0], [ 4.2, -3.0], [ 4.0, -2.0], [ 2.5, -1.5],
            [ 4.0, -0.7]]
goal_set_length = len(goal_set)
# build goal set, where the location of each goal needs to be 0.5m far from obstacles, 
# so as to enable all goals accessible

# lase scan parameters
# number of all laser beams
n_all_laser = 901
laser_angle_resolute = np.pi / (n_all_laser - 1)
# number of used laser beams
n_used_laser = 37
laser_interval = int((n_all_laser - 1) / (n_used_laser - 1))
# in Gazebo, the actual minimum laser range is 0.25m
# here, a little larger range is chosen for detecting collision
laser_min_range = 0.27
# lase scan parameters

# used for visualizing goal
goal_model_dir = '/home/zw/RL_navigation/src/rl_navigation/urdf/Target/model.sdf'
# used for visualizing goal

class Env:
    def __init__(self):
        # used to reset robot state       
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'turtlebot3_burger'
        self.model_state_req.model_state.pose.position.x = 0.0
        self.model_state_req.model_state.pose.position.y = 0.0
        self.model_state_req.model_state.pose.position.z = 0.0
        self.model_state_req.model_state.pose.orientation.x = 0.0
        self.model_state_req.model_state.pose.orientation.y = 0.0
        self.model_state_req.model_state.pose.orientation.z = 0.0
        self.model_state_req.model_state.pose.orientation.w = 1.0   
        self.model_state_req.model_state.twist.linear.x = 0.0
        self.model_state_req.model_state.twist.linear.y = 0.0
        self.model_state_req.model_state.twist.linear.z = 0.0
        self.model_state_req.model_state.twist.angular.x = 0.0
        self.model_state_req.model_state.twist.angular.y = 0.0
        self.model_state_req.model_state.twist.angular.z = 0.0
        self.model_state_req.model_state.reference_frame = 'world'  

        self.joint_name_lst = ['wheel_left_joint', 'wheel_right_joint']
        self.starting_pos = np.array([0.0, 0.0])
        self.model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'turtlebot3_burger'
        self.model_config_req.urdf_param_name = 'robot_description'
        self.model_config_req.joint_names = self.joint_name_lst
        self.model_config_req.joint_positions = self.starting_pos
        # used to reset robot state

        # used to pause and unpause simulation
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # used to pause and unpause simulation

        # visualize goal in Gazebo
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        # visualize goal in Gazebo

        # robot pose
        self.robot_pose = np.zeros(3)
        # relative pose of goal in robot frame
        self.rel_theta = 0.
        self.rel_dis = 0.
        self.rel_dis_last = 0.

        # avoid disorder of data in different threadings
        self.state_lock = threading.Lock()
        # avoid disorder of data in different threadings

        # laser scan overservation for RL
        self.used_scan = np.zeros(n_used_laser, dtype=np.float64)
        # laser scan overservation for RL

        
        # goal position, x, y, which is randomly set
        self.goal_position = np.zeros(2)
        self.goal_position_pose = Pose()
        
        self.goal_count = 0
        self.goal_lock = threading.Lock()
        self.pose = PoseStamped()
        self.pose.header.frame_id = "map"
        self.pose.pose.position.x = 0.
        self.pose.pose.position.y = 0.
        self.pose.pose.position.z = 0.
        self.pose.pose.orientation.x = 0.
        self.pose.pose.orientation.y = 0.
        self.pose.pose.orientation.z = 0.
        self.pose.pose.orientation.w = 1
        
        # if the distance from the robot to the goal is less than threshold_arrive,
        # then the robot successfully arrives the target
        self.threshold_arrive = 0.1
        # used to record the minimum value of all laser beams
        self.min_scan_range = laser_min_range
       
        # motion parameters
        self.look_ahead = 10.0
        # control gain for angular velocity
        self.angle_to_velocity = 0.8
        self.max_angular_velocity = 0.6
        self.max_linear_velocity = 0.2
        self.control_frequency = 10
        self.dt = 1.0 / self.control_frequency
        self.current_step = 0
        self.linear_v_current = 0.
        self.angular_v_current = 0.
        self.linear_v_last = 0.0
        self.angular_v_last = 0.0
        self.linear_acc_max = 0.3
        self.angular_acc_max = 0.4
        # motion parameters

        # subscribe and publish
        self.sub_robot_states = rospy.Subscriber('/robot/states', Float32MultiArray, \
                            self.StateCallback, queue_size=5)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # subscribe and publish

    def StateCallback(self, msg):
        dx = self.goal_position[0] - msg.data[0]
        dy = self.goal_position[1] - msg.data[1]
        rel_dis = hypot(dx, dy)
        rel_theta = atan2(dy, dx) - msg.data[2]
        if rel_theta < -np.pi:
            rel_theta += (2 * np.pi)
        if rel_theta > np.pi:
            rel_theta -= (2 * np.pi)
        self.state_lock.acquire()
        self.robot_pose[0] = msg.data[0]
        self.robot_pose[1] = msg.data[1]
        self.robot_pose[2] = msg.data[2]
        self.linear_v_current = msg.data[3]
        self.angular_v_current = msg.data[4]
        self.rel_dis = rel_dis
        self.rel_theta = rel_theta
        self.state_lock.release()           

    def getState(self, scan):
        scan_range = []
        done = False
        arrive = False

        linear_v = self.linear_v_current
        angular_v = self.angular_v_current
        rel_dis = self.rel_dis

        self.state_lock.acquire()
        self.min_scan_range = min(scan.ranges)
        self.state_lock.release()
        
        if (laser_min_range >= self.min_scan_range):
            done = True
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] > 1.5:
                scan_range.append(1.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] < laser_min_range:
                scan_range.append(laser_min_range)
            else:
                scan_range.append(scan.ranges[i])

        scan_range_reduce = []
        j = 0
        # original scan: 901 dimensions
        # reduced scan: 37 dimnensions
        while (j < len(scan_range)):
            scan_range_reduce.append(scan_range[j])
            j = j + laser_interval
        
        if rel_dis <= self.threshold_arrive or self.current_step == 400:
            arrive = True

        return (np.array(scan_range_reduce), linear_v, angular_v, done, arrive)

    def step(self, action):
        reward = 0.0
        # linear v  [0.00, 0.05, 0.10, 0.15, 0.20] m/s
        # angular v [-90, -60, -30, 0, 30, 60, 90] degree/s
        action_linear = int(action / 7)
        action_angular = action % 7
        if action_linear == 0:
            linear_v = 0.0
            reward -= 0.4
        if action_linear == 1:
            linear_v = 0.05
            reward -= 0.2
        if action_linear == 2:
            linear_v = 0.1
        if action_linear == 3:
            linear_v = 0.15
        if action_linear == 4:
            linear_v = 0.2

        linear_v = action_linear * 0.05
        angular_v = (action_angular - 3) * np.pi / 6.0

        # publish velocity commands
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_v
        vel_cmd.angular.z = angular_v
        self.pub_cmd_vel.publish(vel_cmd)
        # publish velocity commands

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        done, arrive = False, False
        self.used_scan, linear_v, angular_v, done, arrive = self.getState(data)
        if self.min_scan_range >= laser_min_range:
            reward = reward - 0.05 / self.min_scan_range * laser_min_range
        rel_dis_delta = self.rel_dis_last - self.rel_dis
        reward = reward + 180.0 * rel_dis_delta
        self.rel_dis_last = self.rel_dis
        self.linear_v_last = linear_v
        self.angular_v_last = angular_v

        if arrive:
            # print('target arrived at: ', self.current_step)
            self.current_step = 0

        if done:
            # print('collision happened at: ', self.current_step)
            reward = -200
            self.current_step = 0 
        self.current_step = self.current_step + 1

        state1 = self.used_scan / 1.5
        state2 = np.array([self.rel_dis / self.look_ahead, self.rel_theta / np.pi, linear_v, angular_v])

        return state1, state2, reward, done, arrive

    def reset(self, episode):
        # randomly set goal position and robot states
        index_goal_robot = np.random.randint(0, goal_set_length, size=2)
        index_goal = index_goal_robot[0]
        index_robot = index_goal_robot[1]
        while(index_goal == index_robot):
            index_goal_robot = np.random.randint(0, goal_set_length, size=2)
            index_goal = index_goal_robot[0]
            index_robot = index_goal_robot[1]
        self.goal_lock.acquire()
        self.goal_position[0] = goal_set[index_goal][0]
        self.goal_position[1] = goal_set[index_goal][1]
        self.goal_lock.release()
        robot_yaw = np.random.uniform(-1.0, 1.0, 1)
        self.model_state_req.model_state.pose.position.x = goal_set[index_robot][0]
        self.model_state_req.model_state.pose.position.y = goal_set[index_robot][1]
        z = sin(robot_yaw[0] / 2.0)
        w = cos(robot_yaw[0] / 2.0)
        self.model_state_req.model_state.pose.orientation.z = z
        self.model_state_req.model_state.pose.orientation.w = w
        self.reset_robot_state(episode)
        # randomly set goal position and robot states

        rospy.sleep(1.0)
        
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        done, arrive = False, False
        self.used_scan, linear_v, angular_v, done, arrive = self.getState(data)
        self.rel_dis_last = self.rel_dis
        self.linear_v_last = linear_v
        self.angular_v_last = angular_v
        
        state1 = self.used_scan / 1.5
        state2 = np.array([self.rel_dis / self.look_ahead, self.rel_theta / np.pi, linear_v, angular_v])

        return state1, state2

    def reset_robot_state(self, episode):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')

        # if episode > 0:
        #     rospy.wait_for_service('/gazebo/delete_model')
        #     self.del_model('target')

        # Build the target
        # rospy.wait_for_service('/gazebo/spawn_sdf_model')
        # try:
        #     goal_urdf = open(goal_model_dir, "r").read()
        #     target = SpawnModel
        #     target.model_name = 'target'  # the same with sdf name
        #     target.model_xml = goal_urdf
        #     self.goal_position_pose.position.x = self.goal_position[0]
        #     self.goal_position_pose.position.y = self.goal_position[1]
        #     self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position_pose, 'world')
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/failed to build the target")

        #set models pos from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_state call failed')

        #set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_configuration call failed')
     
        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException:
            print('/gazebo/unpause_physics service call failed')
