#!/usr/bin/env python3
import os
import time
import rospy
import argparse
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Reinforcement Learning-based Hierarchical Control 
# for Mapless Navigation of a Wheeled Bipdal Robot 

tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--replay_start', type=int, default=5000)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.998)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.0001)
parser.add_argument('--state1_dim', type=int, default=37)
parser.add_argument('--state2_dim', type=int, default=4)
parser.add_argument('--action_dim', type=int, default=5)


args = parser.parse_args()


class ActionStateModel:
    def __init__(self, state1_dim, state2_dim, action_dim):
        self.state1_dim  = state1_dim
        self.state2_dim  = state2_dim
        self.action_dim = action_dim
        
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
    
    def get_action(self, state1, state2):
        state1 = np.reshape(state1, [1, self.state1_dim])
        state2 = np.reshape(state2, [1, self.state2_dim])
        q_value = self.predict(state1, state2)[0]
        return np.argmax(q_value)

class Agent:
    def __init__(self, state1_dim=args.state1_dim, state2_dim=args.state2_dim, action_dim=args.action_dim):
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.action_dim = action_dim

        self.model = ActionStateModel(self.state1_dim, self.state2_dim, self.action_dim)
        fn = 'model/navigation_5_goal/high_dqn_time_step_1040000.h5'
        self.load_model(fn)       

    def load_model(self, fn):
        self.model.model.load_weights(fn)

    
    
