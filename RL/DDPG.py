#!/usr/bin/env python3
import rospy
import numpy as np
import random
from ddpg_network import *
from environment_ddpg import Env


######## compare work ###########
# Virtual-to-real Deep Reinforcement Learning:
# Continuous Control of Mobile Robots for Mapless Navigation
# 2017, IROS
######## compare work ###########


exploration_decay_start_step = 5000
state1_dim = 10  # laser
state2_dim = 4   # target position relative to the robot frame, last velocity
action_dim = 2   # linear and angular velocities


def main():
    rospy.init_node('ddpg')
    env = Env()
    agent = DDPG(state1_dim + state2_dim, action_dim, replay_start_size=exploration_decay_start_step)

    time_step = 0
    var = 1.0
    eps_decay = 0.9975
    max_episodes = 5000
    data_save = np.zeros((max_episodes, 3), dtype=np.float64)
    print('training start: ', time.time())
    for ep in range(max_episodes):
        done, arrive, total_reward, episode_step =False, False, 0, 0
        state = env.reset(ep)
        while ((not done) and (not arrive)):
            action = agent.act(state)
            if time_step % 1000 == 0 and time_step > exploration_decay_start_step:
                var *= eps_decay
                print('epsilon: ', var)
            a0_var = np.clip(np.random.normal(action[0], var, 1), -1.0, 1.0)
            a1_var = np.clip(np.random.normal(action[1], var, 1), -1.0, 1.0)
            action = np.hstack((a0_var, a1_var))
            next_state, reward, done, arrive, arrive2 = env.step(action)
            if arrive2:
                state = next_state
                continue
            agent.perceive(state, action, reward*0.01, next_state, done)
            total_reward += reward
            state = next_state
           
            if time_step % 40000 == 0 and time_step > 0:
                a_fn = "model/ddpg/ddpg_actor_time_step_{}.h5".format(time_step)
                c_fn = "model/ddpg/ddpg_critic_time_step_{}.h5".format(time_step)
                agent.save_model(a_fn, c_fn)
            time_step = time_step + 1
            episode_step = episode_step + 1
        data_save[ep, 0] = total_reward
        data_save[ep, 1] = episode_step
        if arrive:
            data_save[ep, 2] = 1
        if ep % 100 == 0 and ep > 0:
            fn = 'model/ddpg/ddpg_episode_' + str(ep) + '.txt'
            np.savetxt(fn, data_save)
        if done:
            print('Episode: ', ep, '; Collision at: ', episode_step, '; Average reward: ', total_reward / episode_step)
        if arrive:
            print('Episode: ', ep, '; Arrival at: ', episode_step, '; Average reward: ', total_reward / episode_step)
    print('ending time: ', time.time())
            

if __name__ == '__main__':
     main()
