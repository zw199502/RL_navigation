#!/usr/bin/env python

from io import StringIO
from rosgraph.network import read_ros_handshake_header
import rospy
# Brings in the SimpleActionClient
import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import threading
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from rospy.core import is_shutdown
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
from std_msgs.msg import String

#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, atan2, sqrt, fabs, hypot

# build goal set, where the location of each goal needs to be 0.5m far from obstacles, 
# so as to enable all goals accessible
# map one
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
# map two
# goal_set = [[ 0.0,  0.0], [ 0.5,  0.0], [ 1.5,  0.0], [ 2.5,  0.0], [ 4.2,  0.0], 
#             [ 0.5,  2.0], [ 1.0,  2.0], [ 1.0,  2.5], [ 0.3,  4.0], [ 2.2,  4.2],
#             [ 3.5,  4.2], [ 4.2,  3.2], [ 2.8,  2.6], [ 2.8,  1.6], [ 2.0,  1.2],
#             [ 4.2,  0.9], [-0.5,  0.8], [-1.2,  0.8], [-2.0,  1.0], [-2.0,  1.6],
#             [-2.0,  2.3], [-2.6,  1.3], [-3.0,  1.0], [-2.5,  2.7], [-4.2,  0.5],
#             [-3.5,  2.5], [-4.2,  3.0], [-4.2,  4.2], [-3.0,  4.3], [-1.0,  4.2],
#             [-1.0,  3.3], [-0.2, -0.8], [-1.0, -0.7], [-2.2, -0.7], [-2.5, -1.0],
#             [-0.2, -1.2], [-2.5, -2.5], [-3.6, -2.5], [-4.2, -0.5], [-4.4, -1.5],
#             [ 4.2, -2.0], [-1.7, -2.6], [-1.7, -3.2], [-1.7, -4.2], [-2.8, -4.2],
#             [-1.0, -3.0], [-4.2, -4.2], [ 1.2, -0.8], [ 0.5, -0.5], [ 1.5, -0.5],
#             [ 2.5, -0.5], [ 2.5, -0.8], [ 3.0, -1.0], [ 1.2, -4.0], [ 2.0, -3.8],
#             [ 4.2, -0.5], [ 4.0, -1.5], [ 3.5, -2.5], [ 2.0, -2.5], [ 1.5, -2.5],
#             [ 1.0, -2.5], [ 4.2, -4.2], [ 2.0, -4.0], [ 0.5, -3.0], [ 3.5, -4.2]]
# map three
# goal_set = [[ 0.0,  0.0], [ 0.6,  0.6], [ 0.0,  2.0], [ 0.8,  2.7], [ 1.6,  2.4], 
#             [ 0.5,  4.2], [ 1.5,  4.2], [ 2.8,  4.2], [ 2.5,  0.0], [ 2.6,  1.0],
#             [ 3.3,  0.8], [ 4.0,  1.0], [ 4.2,  2.0], [ 4.2,  3.7], [-1.0,  0.2], 
#             [-2.0,  0.2], [-3.0,  1.0], [-4.0,  1.0], [-2.5,  2.3], [-0.5,  2.2],
#             [-0.3,  0.3], [-2.5,  3.0], [-4.2,  4.2], [-3.5,  4.3], [-2.8,  4.2],
#             [-0.5,  2.3], [-0.6, -0.6], [-1.0, -0.2], [-1.5, -0.5], [-3.7, -0.5],
#             [-4.0, -1.0], [-3.0, -1.6], [-2.0, -2.0], [-1.3, -2.3], [-0.0, -1.0],
#             [-1.2, -2.8], [-3.0, -3.0], [-4.2, -4.2], [-3.0, -4.2], [-2.0, -4.0],
#             [-1.0, -4.4], [ 0.0, -4.2], [ 0.5, -2.1], [ 0.5, -4.3], [ 1.0, -2.0],
#             [ 1.1, -3.2], [ 2.0, -1.0], [ 2.0, -1.5], [ 2.0, -4.0], [ 2.5, -1.5],
#             [ 3.0, -4.1], [ 4.0, -0.5], [ 3.5, -2.5], [ 4.2, -3.0], [ 4.2, -4.2],
#             [ 4.2, -3.5]]
goal_set_length = len(goal_set)
# build goal set, where the location of each goal needs to be 0.5m far from obstacles, 
# so as to enable all goals accessible

# lase scan parameters
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

        # initial states, robot pose and position
        self.initial_states = np.zeros(5)
        # robot pose
        self.robot_pose = np.zeros(3)

        # avoid disorder of data in different threadings
        self.state_lock = threading.Lock()
        # avoid disorder of data in different threadings

        
        # goal position, x, y, which is randomly set
        self.goal_position = np.zeros(2)
        self.goal_position_pose = Pose()
        
        self.goal_count = 0
        # relative distance from the goal to the robot
        self.rel_dis_final_goal = 0.
        
        # if the distance from the robot to the goal is less than threshold_arrive,
        # then the robot successfully arrives the target
        self.threshold_arrive = 0.15
        # used to record the minimum value of all laser beams
        self.min_scan_range = laser_min_range

        # subscribe and publish
        self.sub_robot_states = rospy.Subscriber('/robot/states', Float32MultiArray, \
                            self.StateCallback, queue_size=5)
        # subscribe and publish

        # whether the robot reaches the goal
        self.sub_is_goal_reached = rospy.Subscriber('/if_goal_reached', String, \
                        self.IsGoalReachedCallback, queue_size=5)
        self.is_goal_reached_lock = threading.Lock()
        self.is_goal_reached = False
        # whether the robot reaches the goal

        # Create an action client called "move_base" with action definition file "MoveBaseAction"
        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        # publish zero velocity
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        
    def movebase_client(self):
    
        # Waits until the action server has started up and started listening for goals.
        self.client.wait_for_server()

        # Creates a new goal with the MoveBaseGoal constructor
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        # Move 0.5 meters forward along the x axis of the "map" coordinate frame 
        goal.target_pose.pose.position.x = self.goal_position[0]
        goal.target_pose.pose.position.y = self.goal_position[1]
        # No rotation of the mobile base frame w.r.t. map frame
        goal.target_pose.pose.orientation.w = 1.0

        # Sends the goal to the action server.
        self.client.send_goal(goal)

    def IsGoalReachedCallback(self, msg):
        if msg.data == 'true':
            self.is_goal_reached_lock.acquire()
            self.is_goal_reached = True
            self.is_goal_reached_lock.release()

    def StateCallback(self, msg):
        rel_dis_final_goal = hypot(self.goal_position[0] - msg.data[0], \
                                   self.goal_position[1] - msg.data[1])

        self.state_lock.acquire()
        self.robot_pose[0] = msg.data[0]
        self.robot_pose[1] = msg.data[1]
        self.robot_pose[2] = msg.data[2]
        self.rel_dis_final_goal = rel_dis_final_goal
        self.state_lock.release()           

    def getState(self, scan):
        done = False
        arrive = False
        current_robot_pose = self.robot_pose
        rel_dis_final_goal = self.rel_dis_final_goal
        self.min_scan_range = min(scan.ranges)

        if laser_min_range >= self.min_scan_range:
            done = True
        if self.is_goal_reached == True:
            self.is_goal_reached_lock.acquire()
            self.is_goal_reached = False
            self.is_goal_reached_lock.release()
            arrive = True
        if rel_dis_final_goal <= self.threshold_arrive:
            arrive = True

        return done, arrive, current_robot_pose

    def step(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        # t2 = time.time()
        # print(t2-t1)
        # it continues 0.1s here to get laser data
        done, arrive = False, False
        done, arrive, current_robot_pose = self.getState(data)
        
        if arrive:
            print('target arrived at: ', self.current_step)
            self.current_step = 0

        if done:
            print('collision happened at: ', self.current_step)
            self.current_step = 0 
        self.current_step = self.current_step + 1

        return done, arrive, current_robot_pose

    def reset(self):
        # randomlt set goal position and robot states
        index_goal_robot = np.random.randint(0, goal_set_length, size=2)
        index_goal = index_goal_robot[0]
        index_robot = index_goal_robot[1]
        while(index_goal == index_robot):
            index_goal_robot = np.random.randint(0, goal_set_length, size=2)
            index_goal = index_goal_robot[0]
            index_robot = index_goal_robot[1]
        self.goal_position[0] = goal_set[index_goal][0]
        self.goal_position[1] = goal_set[index_goal][1]
        print(self.goal_position)
        robot_yaw = np.random.uniform(-1.0, 1.0, 1)
        self.model_state_req.model_state.pose.position.x = goal_set[index_robot][0]
        self.model_state_req.model_state.pose.position.y = goal_set[index_robot][1]
        z = sin(robot_yaw[0] / 2.0)
        w = cos(robot_yaw[0] / 2.0)
        self.model_state_req.model_state.pose.orientation.z = z
        self.model_state_req.model_state.pose.orientation.w = w
        self.reset_robot_state()
        # randomlt set goal position and robot states
        self.current_step = 0
        rospy.sleep(2.0)
        
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        done, arrive, current_robot_pose = self.getState(data)

        self.initial_states = np.hstack((current_robot_pose, self.goal_position))
       
        return done, arrive, current_robot_pose

    def reset_robot_state(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')

        # visualize goal
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
        # visualize goal

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

    def show_all_goal(self):
        for i in range(goal_set_length):
            # visualize goal
            if i > 0:
                rospy.wait_for_service('/gazebo/delete_model')
                self.del_model('target')

            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                self.goal_position[0] = goal_set[i][0]
                self.goal_position[1] = goal_set[i][1]
                print(self.goal_position)
                self.goal_position_pose.position.x = self.goal_position[0]
                self.goal_position_pose.position.y = self.goal_position[1]
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position_pose, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.sleep(2)
            # visualize goal

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('dwa_test')
    dwa_env = Env()
    # dwa_env.show_all_goal()
    path_record = []
    done, arrive, current_robot_pose = dwa_env.reset()
    print('initial states:')
    print(dwa_env.initial_states)
    current_robot_pose_list = current_robot_pose.tolist()
    path_record.append(current_robot_pose_list)
    try:
        dwa_env.movebase_client()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
    reset_time = 4
    max_step = 2000
    count_step = 0
    
    while(not rospy.is_shutdown()):
        done, arrive, current_robot_pose = dwa_env.step()
        current_robot_pose_list = current_robot_pose.tolist()
        path_record.append(current_robot_pose_list)
        if done or arrive:
            fn = "model/dwa_test/unknown/path_{}.txt".format(reset_time)
            path_record_numpy = np.array(path_record)
            np.savetxt(fn, path_record_numpy)
            break
        
        count_step = count_step + 1
        if count_step == max_step:
            print('max step arrived')
            dwa_env.current_step = 0
            fn = "model/dwa_test/unknown/path_{}.txt".format(reset_time)
            path_record_numpy = np.array(path_record)
            np.savetxt(fn, path_record_numpy)
            break
    print('over')



