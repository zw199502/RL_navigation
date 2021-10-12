#!/usr/bin/env python3
import os
import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from tensorflow.keras.optimizers import Adam

from std_srvs.srv import Empty
import rospy


from Prioritized_Replay import Memory

# Original paper: https://arxiv.org/pdf/1509.02971.pdf
# DDPG with PER paper: https://cardwing.github.io/files/RL_course_report.pdf

tf.keras.backend.set_floatx('float32')

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], \
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

def actor(state_shape):
    
    state_input = Input((state_shape,), name='policy_state2_input')

    linear1 = Dense(512, activation='relu', name='policy1')(state_input)
    linear2 = Dense(512,  activation='relu', name='policy2')(linear1)
    linear3 = Dense(512,  activation='relu', name='policy3')(linear2)

    net_output = Dense(2, name="Out", activation='tanh')(linear3)
    
    model = Model(state_input, net_output)

    return model


def critic(state_shape, action_dim):
   
    state_input = Input((state_shape,), name='q_state_input')
    dense1 = Dense(512, activation='relu', name='policy_dense1')(state_input)

    action_input = Input((action_dim,), name='q_action_input')

    connect1 = concatenate([dense1, action_input], axis=-1)
    linear1 = Dense(512, activation='relu',   name='q1')(connect1)
    linear2 = Dense(512,  activation='relu',   name='q2')(linear1)
    out_q = Dense(1,   activation='linear', name='q3')(linear2)
    return Model([state_input, action_input], out_q)


def update_target_weights(model, target_model, tau=0.0001):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)

class DDPG:
    def __init__(
            self,
            state_dim,
            action_dim,
            use_priority=False,
            lr_actor=1e-5,
            lr_critic=1e-5,
            tau=0.0001,
            gamma=0.99,
            batch_size=64,
            iteration_update=10,
            replay_start_size=5000,
            memory_cap=50000
    ):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.state_shape = state_dim 
        self.action_dim = action_dim  # number of actions

        self.use_priority = use_priority
        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)
        self.replay_start_size = replay_start_size
       
        # Define and initialize Actor network
        self.actor = actor(self.state_shape)
        self.actor_target = actor(self.state_shape)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau)

        # Define and initialize Critic network
        self.critic = critic(self.state_shape, self.action_dim)
        self.critic_target = critic(self.state_shape, self.action_dim)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic, self.critic_target, tau)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size
        self.iteration_update = iteration_update

    def act(self, inputs):
        inputs = np.reshape(inputs, [1, self.state_shape])
        a = self.actor.predict(inputs)[0]
        return a

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
  

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
  
    def remember(self, state, action, reward, next_state, done):
        if self.use_priority:
            action = np.squeeze(action)
            transition = np.hstack([state, action, reward, next_state, done])
            self.memory.store(transition)
        else:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        
        if self.use_priority:
            tree_idx, samples, ISWeights = self.memory.sample(self.batch_size)
            split_shape = np.cumsum([self.state_shape, self.action_dim, 1, self.state_shape])
            states, actions, rewards, next_states, dones = np.hsplit(samples, split_shape)
        else:
            ISWeights = 1.0
            samples = random.sample(self.memory, self.batch_size)
            s = np.array(samples).T
            states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]

        next_actions = self.actor_target.predict(next_states)
        q_future = self.critic_target.predict([next_states, next_actions])
        target_qs = rewards + q_future * self.gamma * (1. - dones)

        # train critic
        with tf.GradientTape() as tape:
            states = np.float32(states)
            actions = np.float32(actions)
            q_values = self.critic([states, actions])
            td_error = q_values - target_qs
            critic_loss = tf.reduce_mean(ISWeights * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)  # compute critic gradient
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # update priority
        if self.use_priority:
            abs_errors = tf.reduce_sum(tf.abs(td_error), axis=1)
            self.memory.batch_update(tree_idx, abs_errors)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        
        
    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.remember(state, action, reward, next_state, done)  # add to memory
        if len(self.memory) == self.replay_start_size:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if len(self.memory) >= self.replay_start_size:
            
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                self.pause()
            except (rospy.ServiceException, e):
                print ("/gazebo/pause_physics service call failed")

            for i in range(self.iteration_update):
                self.replay()
            update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
            update_target_weights(self.critic, self.critic_target, tau=self.tau)
                
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except (rospy.ServiceException, e):
                print ("/gazebo/unpause_physics service call failed")
        

