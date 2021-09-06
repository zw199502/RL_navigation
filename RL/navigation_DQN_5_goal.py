#!/usr/bin/env python3
import os
import time
import argparse
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
import rospy
from std_srvs.srv import Empty
from navigation_environment_5_goal import Env

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], \
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

# Reinforcement Learning-based Hierarchical Control 
# for Mapless Navigation of a Wheeled Bipdal Robot 

tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--replay_start', type=int, default=5000)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9975)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.0001)
parser.add_argument('--state1_dim', type=int, default=37)
parser.add_argument('--state2_dim', type=int, default=4)
parser.add_argument('--action_dim', type=int, default=2)


args = parser.parse_args()

max_episodes=5000

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state1, state2, action, reward, next_state1, next_state2, done):
        self.buffer.append([state1, state2, action, reward, next_state1, next_state2, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states1, states2, actions, rewards, next_states1, next_states2, done = map(np.asarray, zip(*sample))
        states1 = np.array(states1).reshape(args.batch_size, -1)
        states2 = np.array(states2).reshape(args.batch_size, -1)
        next_states1 = np.array(next_states1).reshape(args.batch_size, -1)
        next_states2 = np.array(next_states2).reshape(args.batch_size, -1)
        return states1, states2, actions, rewards, next_states1, next_states2, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state1_dim, state2_dim, action_dim):
        self.state1_dim  = state1_dim
        self.state2_dim  = state2_dim
        self.action_dim = action_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
    
    def create_model(self):
        state1_input = Input((self.state1_dim,), name='q_state1_input')
        dense1 = Dense(256, activation='relu', name='q_dense1')(state1_input)
        dense2 = Dense(256,  activation='relu', name='q_dense2')(dense1)
        dense3 = Dense(10,  activation='tanh', name='q_dense3')(dense2)
    
        state2_input = Input((self.state2_dim,), name='q_state_input')

        connect1 = concatenate([dense3, state2_input], axis=-1)
        linear1 = Dense(128, activation='relu',   name='q1')(connect1)
        linear2 = Dense(128,  activation='relu',   name='q2')(linear1)
        linear3 = Dense(64,  activation='relu',   name='q3')(linear2)
        out_q = Dense(self.action_dim,   activation='linear', name='q4')(linear3)
        _model = Model([state1_input, state2_input], out_q)
        _model.compile(loss='mse', optimizer=Adam(args.lr))
        return _model
    
    def predict(self, state1, state2):
        return self.model.predict([state1, state2])
    
    def get_action(self, state1, state2, time_step):
        state1 = np.reshape(state1, [1, self.state1_dim])
        state2 = np.reshape(state2, [1, self.state2_dim])
        if time_step % 1000 == 0 and time_step > args.replay_start:
            self.epsilon *= args.eps_decay
            print('epsilon: ', self.epsilon)
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state1, state2)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states1, states2, targets):
        self.model.fit([states1, states2], targets, epochs=1, verbose=0)

    def save(self, fn):
        self.model.save(fn)
    

class Agent:
    def __init__(self, env, state1_dim, state2_dim, action_dim):
        self.env = env
        self.time_step = 0
        self.count_contious_failure = 0
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.action_dim = action_dim

        self.model = ActionStateModel(self.state1_dim, self.state2_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state1_dim, self.state2_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

        # used to pause and unpause simulation
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # used to pause and unpause simulation

    def target_update(self):
        weights = self.model.model.get_weights()
        target_weights = self.target_model.model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * args.tau + target_weights[i] * (1 - args.tau)
        self.target_model.model.set_weights(target_weights)
    
    def replay(self):
        for _ in range(10):
            states1, states2, actions, rewards, next_states1, next_states2, done = self.buffer.sample()
            targets = self.target_model.predict(states1, states2)
            next_q_values = self.target_model.predict(next_states1, next_states2).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states1, states2, targets)
    
    def train(self):
        data_save = np.zeros((max_episodes, 3), dtype=np.float64)
        print('training start: ', time.time())
        for ep in range(max_episodes):
            done, arrive, total_reward, episode_step =False, False, 0, 0
            state1, state2 = self.env.reset(ep)
            while ((not done) and (not arrive)):
                action = self.model.get_action(state1, state2, self.time_step)
                next_state1, next_state2, reward, done, arrive, arrive2 = self.env.step(action)
                if arrive2:
                    state1 = next_state1
                    state2 = next_state2
                    continue
                self.buffer.put(state1, state2, action, reward*0.01, next_state1, next_state2, done)
                total_reward += reward
                state1 = next_state1
                state2 = next_state2
                if self.buffer.size() == args.replay_start:
                    print('updating start: ', time.time())
                   
                if self.buffer.size() >= args.replay_start:
                     # pause Gazebo for update neural networks
                    rospy.wait_for_service('/gazebo/pause_physics')
                    try:
                        self.pause_proxy()
                    except rospy.ServiceException:
                        print('/gazebo/pause_physics service call failed')

                    self.replay()
                    self.target_update()
                    
                    # unpause physics
                    rospy.wait_for_service('/gazebo/unpause_physics')
                    try:
                        self.unpause_proxy()
                    except rospy.ServiceException:
                        print('/gazebo/unpause_physics service call failed')
                if self.time_step % 40000 == 0 and self.time_step > 0:
                    self.save_model("model/navigation_5_goal/dqn_time_step_{}.h5".format(self.time_step))
                self.time_step = self.time_step + 1
                episode_step = episode_step + 1
            data_save[ep, 0] = total_reward
            data_save[ep, 1] = episode_step
            if arrive:
                data_save[ep, 2] = 1 
            if ep % 100 == 0 and ep > 0:
                fn = 'model/navigation_5_goal/dqn_episode_' + str(ep) + '.txt'
                np.savetxt(fn, data_save)
            if done:
                print('Episode: ', ep, '; Collision at: ', episode_step, '; Average reward: ', total_reward / episode_step)
            if arrive:
                print('Episode: ', ep, '; Arrival at: ', episode_step, '; Average reward: ', total_reward / episode_step)
        print('ending time: ', time.time())

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model.model.load_weights(fn)

    
if __name__=='__main__':
    rospy.init_node('rl_navigation')
    _env = Env()
    _agent = Agent(_env, args.state1_dim, args.state2_dim, args.action_dim)
    _agent.train()
