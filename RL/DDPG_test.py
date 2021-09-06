#!/usr/bin/env python3
import rospy
import numpy as np
import random
from ddpg_network import *
from environment_ddpg_test import Env

exploration_decay_start_step = 5000
######## compare work ###########
# Virtual-to-real Deep Reinforcement Learning:
# Continuous Control of Mobile Robots for Mapless Navigation
# 2017, IROS
######## compare work ###########


state1_dim = 10  # laser
state2_dim = 4   # target position relative to the robot frame, last velocity
action_dim = 2   # linear and angular velocities


def main():
    rospy.init_node('ddpg')
    env = Env()
    agent = DDPG(state1_dim + state2_dim, action_dim, replay_start_size=exploration_decay_start_step)
    a_fn = 'model/ddpg/version1/ddpg_actor_time_step_800000.h5'
    c_fn = 'model/ddpg/version1/ddpg_critic_time_step_800000.h5'
    agent.load_actor(a_fn)
    agent.load_critic(c_fn)
    # test 100 times
    test_times = 100
    # for ep in range(test_times):
    #     path_record = []
    #     done, arrive, ep_step = False, False, 0
    #     state, current_robot_pose = env.reset(ep)
    #     current_robot_pose_list = current_robot_pose.tolist()
    #     path_record.append(current_robot_pose_list)
    #     while ((not done) and (not arrive)):
    #         action = agent.act(state)
    #         next_state, reward, done, arrive, arrive2, current_robot_pose = env.step(action)
    #         current_robot_pose_list = current_robot_pose.tolist()
    #         path_record.append(current_robot_pose_list)
    #         state = next_state
    #         ep_step = ep_step + 1
    #         if done:
    #             print(0, ep_step)
    #             fn = "data_processing/ddpg/path_{}.txt".format(ep + 1)
    #             path_record_numpy = np.array(path_record)
    #             np.savetxt(fn, path_record_numpy)
    #         if arrive:
    #             print(1, ep_step)
    #             fn = "data_processing/ddpg/path_{}.txt".format(ep + 1)
    #             path_record_numpy = np.array(path_record)
    #             np.savetxt(fn, path_record_numpy)

    # test only once
    ep = 27  # 27, 36, 46, 72, 94
    path_record = []
    done, arrive, ep_step = False, False, 0
    state, current_robot_pose = env.reset(ep)
    current_robot_pose_list = current_robot_pose.tolist()
    path_record.append(current_robot_pose_list)
    while ((not done) and (not arrive)):
        action = agent.act(state)
        next_state, reward, done, arrive, arrive2, current_robot_pose = env.step(action)
        current_robot_pose_list = current_robot_pose.tolist()
        path_record.append(current_robot_pose_list)
        state = next_state
        ep_step = ep_step + 1
    print('over')
           
          
            

if __name__ == '__main__':
     main()
