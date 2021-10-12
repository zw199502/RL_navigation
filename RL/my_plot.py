#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# file_reward = 'model/dqn_episode_260.txt'
# reward = np.loadtxt(file_reward)
# # print(reward[260, :])
# reward_valid = reward[0:261, :]
# reward_valid_average = np.zeros(261, dtype=np.float64)
# for i in range(261):
#     reward_valid_average[i] = reward_valid[i, 0] / reward_valid[i, 1]

# fig, ax1 = plt.subplots()
 
# ax1.plot(reward_valid_average, color='blue', alpha=0.5, label='average reward')
# ax1.set_xlabel("episodes")
# ax1.set_ylabel("average reward")

# plt.show()

file_name = 'model/navigation_5/first/dqn_episode_2900.txt'
data = np.loadtxt(file_name)
length = data.shape[0]
data_average = np.zeros(length)
for i in range(length):
    data_average[i] = data[i, 0] / data[i, 1]
plt.plot(data_average)

plt.show()

