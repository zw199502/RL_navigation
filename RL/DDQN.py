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
from environment_ddqn import Env


################ ddqn compare ####################
# Discrete Deep Reinforcement Learning for Mapless Navigation
# ICRA, 2020
################ ddqn compare ####################


tf.keras.backend.set_floatx('float64')

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], \
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

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

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
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
    
    def get_action(self, state, time_step):
        state = np.reshape(state, [1, self.state_dim])
        if time_step % 1000 == 0 and time_step > args.replay_start:
            self.epsilon *= args.eps_decay
            print('epsilon: ', self.epsilon)
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, fn):
        self.model.save(fn)
    

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 15
        self.action_dim = 5

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.time_step = 0

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model.model.load_weights(fn)

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException, e):
            print ("/gazebo/pause_physics service call failed")

        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException, e):
            print ("/gazebo/unpause_physics service call failed")
    
    def train(self):
        data_save = np.zeros((max_episodes, 3), dtype=np.float64)
        print('training start: ', time.time())
        for ep in range(max_episodes):
            done, arrive, total_reward, episode_step =False, False, 0, 0
            state = self.env.reset(ep)
            while ((not done) and (not arrive)):
                action = self.model.get_action(state, self.time_step)
                next_state, reward, done, arrive, arrive2 = self.env.step(action)
                if arrive2:
                    state = next_state
                    continue
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
                if self.buffer.size() == args.replay_start:
                    print('updating start: ', time.time())
                   
                if self.buffer.size() >= args.replay_start:
                    self.replay()
                    self.target_update()
                if self.time_step % 40000 == 0 and self.time_step > 0:
                    self.save_model("model/ddqn/version3/ddqn_time_step_{}.h5".format(self.time_step))
                self.time_step = self.time_step + 1
                episode_step = episode_step + 1
            data_save[ep, 0] = total_reward
            data_save[ep, 1] = episode_step
            if arrive:
                data_save[ep, 2] = 1
            if ep % 100 == 0 and ep > 0:
                fn = 'model/ddqn/version3/ddqn_episode_' + str(ep) + '.txt'
                np.savetxt(fn, data_save)
            if done:
                print('Episode: ', ep, '; Collision at: ', episode_step, '; Average reward: ', total_reward / episode_step)
            if arrive:
                print('Episode: ', ep, '; Arrival at: ', episode_step, '; Average reward: ', total_reward / episode_step)
        print('ending time: ', time.time())

def main():
    rospy.init_node('ddqn')
    env = Env()
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()