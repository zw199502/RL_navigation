#!/usr/bin/env python3
import rospy
import time
from std_srvs.srv import Empty
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random
from environment_ddqn_test import Env


################ ddqn compare ####################
# Discrete Deep Reinforcement Learning for Mapless Navigation
# ICRA, 2020
################ ddqn compare ####################


tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9975)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--replay_start', type=int, default=5000)

args = parser.parse_args()

max_episodes = 5000


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        
        self.model = self.create_model()
    
    def create_model(self):
        state_input = Input((self.state_dim,))
        backbone_1 = Dense(64, activation='relu')(state_input)
        backbone_2 = Dense(64, activation='relu')(backbone_1)
        value_output = Dense(1)(backbone_2)
        advantage_output = Dense(self.action_dim)(backbone_2)
        output = Add()([value_output, advantage_output])
        model = tf.keras.Model(state_input, output)
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        q_value = self.predict(state)[0]
        return np.argmax(q_value)
    

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 15
        self.action_dim = 5
        self.model = ActionStateModel(self.state_dim, self.action_dim)

    def load_model(self, fn):
        self.model.model.load_weights(fn)
    
    def test(self):
        test_times = 100
        for ep in range(test_times):
            path_record = []
            done, arrive, ep_step = False, False, 0
            state, current_robot_pose = self.env.reset(ep)
            current_robot_pose_list = current_robot_pose.tolist()
            path_record.append(current_robot_pose_list)
            while ((not done) and (not arrive)):
                action = self.model.get_action(state)
                next_state, reward, done, arrive, arrive2, current_robot_pose = self.env.step(action)
                current_robot_pose_list = current_robot_pose.tolist()
                path_record.append(current_robot_pose_list)
                state = next_state
                ep_step = ep_step + 1
                if done:
                    print(0, ep_step)
                    fn = "data_processing/ddqn/path_{}.txt".format(ep + 1)
                    path_record_numpy = np.array(path_record)
                    np.savetxt(fn, path_record_numpy)
                if arrive:
                    print(1, ep_step)
                    fn = "data_processing/ddqn/path_{}.txt".format(ep + 1)
                    path_record_numpy = np.array(path_record)
                    np.savetxt(fn, path_record_numpy)
        print('over')

    def test_once(self):
        path_record = []
        ep = 27  # 27, 36, 46, 72, 94
        done, arrive, ep_step = False, False, 0
        state, current_robot_pose = self.env.reset(ep)
        current_robot_pose_list = current_robot_pose.tolist()
        path_record.append(current_robot_pose_list)
        while ((not done) and (not arrive)):
            action = self.model.get_action(state)
            next_state, reward, done, arrive, arrive2, current_robot_pose = self.env.step(action)
            current_robot_pose_list = current_robot_pose.tolist()
            path_record.append(current_robot_pose_list)
            state = next_state
            ep_step = ep_step + 1
        print('over')

def main():
    rospy.init_node('ddqn')
    _env = Env()
    _agent = Agent(_env)
    fn = 'model/ddqn/version1/ddqn_time_step_840000.h5'
    _agent.load_model(fn)
    # _agent.test()
    _agent.test()

if __name__ == "__main__":
    main()
    