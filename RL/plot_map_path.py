#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
from scipy import signal
from math import hypot

##################   average reward for three DRL framworkds   #########################
length = 1000
x = np.linspace(0, length, num=length)

file_name_dqn1 = 'data_processing/average_reward/v1_dqn_episode_1000.txt'
data_dqn_full1 = np.loadtxt(file_name_dqn1)
data_dqn1 = data_dqn_full1[0:length, :]

file_name_dqn2 = 'data_processing/average_reward/v2_dqn_episode_1000.txt'
data_dqn_full2 = np.loadtxt(file_name_dqn2)
data_dqn2 = data_dqn_full2[0:length, :]

file_name_ddqn1 = 'data_processing/average_reward/v1_ddqn_episode_1000.txt'
data_ddqn_full1 = np.loadtxt(file_name_ddqn1)
data_ddqn1 = data_ddqn_full1[0:length, :]

file_name_ddqn2 = 'data_processing/average_reward/v2_ddqn_episode_1000.txt'
data_ddqn_full2 = np.loadtxt(file_name_ddqn2)
data_ddqn2 = data_ddqn_full2[0:length, :]

file_name_ddpg1 = 'data_processing/average_reward/v1_ddpg_episode_1000.txt'
data_ddpg_full1 = np.loadtxt(file_name_ddpg1)
data_ddpg1 = data_ddpg_full1[0:length, :]

file_name_ddpg2 = 'data_processing/average_reward/v2_ddpg_episode_1000.txt'
data_ddpg_full2 = np.loadtxt(file_name_ddpg2)
data_ddpg2 = data_ddpg_full2[0:length, :]

zero_dqn1 = 0
zero_dqn2 = 0
zero_ddqn1 = 0
zero_ddqn2 = 0
zero_ddpg1 = 0
zero_ddpg2 = 0
for i in range(length):
    if data_dqn1[i, 2] == 0:
        zero_dqn1 = zero_dqn1 + 1
    if data_dqn2[i, 2] == 0:
        zero_dqn2 = zero_dqn2 + 1
    # if data_ddqn1[2] == 0:
    #     zero_ddqn1 = zero_ddqn1 + 1
    if data_ddqn2[i, 2] == 0:
        zero_ddqn2 = zero_ddqn2 + 1
    if data_ddpg1[i, 2] == 0:
        zero_ddpg1 = zero_ddpg1 + 1
    if data_ddpg2[i, 2] == 0:
        zero_ddpg2 = zero_ddpg2 + 1
print((zero_dqn1 + zero_dqn2)/2.0)
print(zero_ddqn2)
# print((zero_ddqn1 + zero_ddqn2)/2.0)
print((zero_ddpg1 + zero_ddpg2)/2.0)

# data_average_dqn1 = np.zeros(length)
# data_average_dqn2 = np.zeros(length)
# data_average_ddqn1 = np.zeros(length)
# data_average_ddqn2 = np.zeros(length)
# data_average_ddpg1 = np.zeros(length)
# data_average_ddpg2 = np.zeros(length)
# for i in range(length):
#     data_average_dqn1[i] = data_dqn1[i, 0] / data_dqn1[i, 1]
#     data_average_dqn2[i] = data_dqn2[i, 0] / data_dqn2[i, 1]
#     data_average_ddqn1[i] = data_ddqn1[i, 0] / data_ddqn1[i, 1]
#     data_average_ddqn2[i] = data_ddqn2[i, 0] / data_ddqn2[i, 1]
#     data_average_ddpg1[i] = data_ddpg1[i, 0] / data_ddpg1[i, 1]
#     data_average_ddpg2[i] = data_ddpg2[i, 0] / data_ddpg2[i, 1]

# b, a = signal.butter(8, [0.2,0.8], 'bandstop')
# filted_data_average_ddqn1 = signal.filtfilt(b, a, data_average_ddqn1)
# filted_data_average_ddqn2 = signal.filtfilt(b, a, data_average_ddqn2)
# data_average_ddqn = np.concatenate((filted_data_average_ddqn1, filted_data_average_ddqn2))
# episode = np.concatenate((x, x))
# sns.lineplot(episode, data_average_ddqn)

# filted_data_average_ddpg1 = signal.filtfilt(b, a, data_average_ddpg1)
# filted_data_average_ddpg2 = signal.filtfilt(b, a, data_average_ddpg2)
# data_average_ddpg = np.concatenate((filted_data_average_ddpg1, filted_data_average_ddpg2))
# episode = np.concatenate((x, x))
# sns.lineplot(episode, data_average_ddpg)

# b, a = signal.butter(8, [0.2,0.8], 'bandstop')
# filted_data_average_dqn1 = signal.filtfilt(b, a, data_average_dqn1)
# filted_data_average_dqn2 = signal.filtfilt(b, a, data_average_dqn2)
# data_average_dqn = np.concatenate((filted_data_average_dqn1, filted_data_average_dqn2))
# episode = np.concatenate((x, x))
# sns.lineplot(episode, data_average_dqn)


# matplotlib.rc('xtick', labelsize=14) 
# matplotlib.rc('ytick', labelsize=14) 
# plt.xlabel('iteration', fontsize=14)
# plt.ylabel('average reward', fontsize=14)
# plt.legend(["DDQN", "DDPG", "DQN"], loc ="lower right", ncol=3)

# plt.show()
##################   average reward for three DRL framworkds   #########################

####################   calculate average time  ##################################
# dqn_log_file = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/log.txt'
# dqn_low_log_file = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/log.txt'
# dwa_log_file = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/log.txt'
# ddpg_log_file = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/log.txt'
# ddqn_log_file = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/log.txt'
# dqn_log = np.loadtxt(dqn_log_file)
# dqn_low_log = np.loadtxt(dqn_low_log_file)
# dwa_log = np.loadtxt(dwa_log_file)
# ddpg_log = np.loadtxt(ddpg_log_file)
# ddqn_log = np.loadtxt(ddqn_log_file)
# length = 100
# one_index = []
# for i in range(100):
#     flag = dqn_log[i, 0] * dqn_low_log[i, 0] * dwa_log[i, 0] * ddpg_log[i, 0] * ddqn_log[i, 0]
#     if flag == 1:
#         one_index.append(i)
# one_index_length = len(one_index)
# average_time_dqn = 0.
# average_time_dqn_low = 0.
# average_time_dwa = 0.
# average_time_ddpg = 0.
# average_time_ddqn = 0.
# for index in one_index:
#     average_time_dqn = average_time_dqn + dqn_log[index, 1]
#     average_time_dqn_low = average_time_dqn_low + dqn_low_log[index, 1]
#     average_time_dwa = average_time_dwa + dwa_log[index, 1]
#     average_time_ddpg = average_time_ddpg + ddpg_log[index, 1]
#     average_time_ddqn = average_time_ddqn + ddqn_log[index, 1]
# print(average_time_dqn / one_index_length / 10.0)
# print(average_time_dqn_low / one_index_length / 10.0)
# print(average_time_dwa / one_index_length / 10.0)
# print(average_time_ddpg / one_index_length / 10.0)
# print(average_time_ddqn / one_index_length / 10.0)
####################   calculate average time  ##################################

####################          comparison in world one       ###########################################
# file_map = '/home/zw/RL_navigation/src/rl_navigation/maps/map_crop.png'
# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/path_94.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/path_27.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/path_72.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/path_36.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/version2/path_46.txt'

# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/path_94.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/path_27.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/path_72.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/path_36.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/dqn_low/version2/path_46.txt'

# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/path_94.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/path_27.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/path_72.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/path_36.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/dwa/version2/path_46.txt'

# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/path_94.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/path_27.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/path_72.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/path_36.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/ddqn/path_46.txt'

# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/path_94.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/path_27.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/path_72.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/path_36.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/ddpg/path_46.txt'

# img = cv2.imread(file_map, cv2.IMREAD_GRAYSCALE)
# resolution = 0.05
# height = img.shape[0]
# width = img.shape[1]
# for i in range(height):
#     img[i, width - 1] = 0
#     img[i, 0] = 0
# for j in range(width):
#     img[height - 1, j] = 0
#     img[0, j] = 0
# point = []
# for i in range(height):
#     for j in range(width):
#         if img[i, j] < 127:
#             x = (j - (width - 1) / 2) * resolution
#             y = -(i - (height - 1) / 2) * resolution
#             point.append([x, y])

# figure(figsize=(8, 8), dpi=80)
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 

# point_array = np.array(point)
# point_array_x = point_array[:, 0]
# point_array_y = point_array[:, 1]
# plt.scatter(point_array_x.T, point_array_y.T, c='k')

# path1 = np.loadtxt(file_path1)
# path2 = np.loadtxt(file_path2)
# path3 = np.loadtxt(file_path3)
# path4 = np.loadtxt(file_path4)
# path5 = np.loadtxt(file_path5)
# a1 = 0
# for i in range(path1.shape[0] - 1):
#     dx = path1[i + 1, 0] - path1[i, 0]
#     dy = path1[i + 1, 1] - path1[i, 1]
#     a1 += hypot(dx, dy)
# print(a1/path1.shape[0])
# a1 = a1/path1.shape[0]
# a2 = 0
# for i in range(path2.shape[0] - 1):
#     dx = path2[i + 1, 0] - path2[i, 0]
#     dy = path2[i + 1, 1] - path2[i, 1]
#     a2 += hypot(dx, dy)
# print(a2/path2.shape[0])
# a2 = a2/path2.shape[0]
# a3 = 0
# for i in range(path3.shape[0] - 1):
#     dx = path3[i + 1, 0] - path3[i, 0]
#     dy = path3[i + 1, 1] - path3[i, 1]
#     a3 += hypot(dx, dy)
# print(a3/path3.shape[0])
# a3 = a3/path3.shape[0]
# a4 = 0
# for i in range(path4.shape[0] - 1):
#     dx = path4[i + 1, 0] - path4[i, 0]
#     dy = path4[i + 1, 1] - path4[i, 1]
#     a4 += hypot(dx, dy)
# print(a4/path4.shape[0])
# a4 = a4/path4.shape[0]
# a5 = 0
# for i in range(path5.shape[0] - 1):
#     dx = path5[i + 1, 0] - path5[i, 0]
#     dy = path5[i + 1, 1] - path5[i, 1]
#     a5 += hypot(dx, dy)
# print(a5/path5.shape[0])
# a5 = a5/path5.shape[0]
# print((a1+a2+a3+a4+a5)/5.0)

# x = path1[:, 0]
# y = path1[:, 1]
# plt.plot(x.T, y.T, label='P1')

# x = path2[:, 0]
# y = path2[:, 1]
# plt.plot(x.T, y.T, label='P2')

# x = path3[:, 0]
# y = path3[:, 1]
# plt.plot(x.T, y.T, label='P3')

# x = path4[:, 0]
# y = path4[:, 1]
# plt.plot(x.T, y.T, label='P4')

# x = path5[:, 0]
# y = path5[:, 1]
# plt.plot(x.T, y.T, label='P5')

# plt.legend(bbox_to_anchor = (1.0, 1.09), ncol = 5, fontsize=15)

# plt.xlabel('x/m', fontsize=20)
# plt.ylabel('y/m', fontsize=20)
# plt.show()
####################          comparison in world one       ###########################################



####################          test in world two and three       ###########################################
# file_map = '/home/zw/RL_navigation/src/rl_navigation/maps/map3_crop.pgm'
# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_5.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_6.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_8.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_21.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_18.txt'
# file_path6 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_27.txt'
# file_path7 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_31.txt'
# file_path8 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_46.txt'
# file_path9 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_57.txt'
# file_path10 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world3/version2/path_94.txt'

# file_map = '/home/zw/RL_navigation/src/rl_navigation/maps/map2_crop.pgm'
# file_path1 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_2.txt'
# file_path2 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_3.txt'
# file_path3 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_11.txt'
# file_path4 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_18.txt'
# file_path5 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_21.txt'
# file_path6 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_31.txt'
# file_path7 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_50.txt'
# file_path8 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_69.txt'
# file_path9 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_84.txt'
# file_path10 = '/home/zw/RL_navigation/src/RL/data_processing/dqn/world2/version2/path_94.txt'

# img = cv2.imread(file_map, cv2.IMREAD_GRAYSCALE)
# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# resolution = 0.05
# height = img.shape[0]
# width = img.shape[1]
# for i in range(height):
#     img[i, width - 1] = 0
#     img[i, 0] = 0
# for j in range(width):
#     img[height - 1, j] = 0
#     img[0, j] = 0
# point = []
# for i in range(height):
#     for j in range(width):
#         if img[i, j] < 127:
#             x = (j - (width - 1) / 2) * resolution
#             y = -(i - (height - 1) / 2) * resolution
#             point.append([x, y])

# figure(figsize=(8, 8), dpi=80)
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 

# point_array = np.array(point)
# point_array_x = point_array[:, 0]
# point_array_y = point_array[:, 1]
# plt.scatter(point_array_x.T, point_array_y.T, c='k')

# path1 = np.loadtxt(file_path1)
# path2 = np.loadtxt(file_path2)
# path3 = np.loadtxt(file_path3)
# path4 = np.loadtxt(file_path4)
# path5 = np.loadtxt(file_path5)
# path6 = np.loadtxt(file_path6)
# path7 = np.loadtxt(file_path7)
# path8 = np.loadtxt(file_path8)
# path9 = np.loadtxt(file_path9)
# path10 = np.loadtxt(file_path10)

# x = path1[:, 0]
# y = path1[:, 1]
# plt.plot(x.T, y.T)

# x = path2[:, 0]
# y = path2[:, 1]
# plt.plot(x.T, y.T)

# x = path3[:, 0]
# y = path3[:, 1]
# plt.plot(x.T, y.T)

# x = path4[:, 0]
# y = path4[:, 1]
# plt.plot(x.T, y.T)

# x = path5[:, 0]
# y = path5[:, 1]
# plt.plot(x.T, y.T)

# x = path6[:, 0]
# y = path6[:, 1]
# plt.plot(x.T, y.T)

# x = path7[:, 0]
# y = path7[:, 1]
# plt.plot(x.T, y.T)

# x = path8[:, 0]
# y = path8[:, 1]
# plt.plot(x.T, y.T)

# x = path9[:, 0]
# y = path9[:, 1]
# plt.plot(x.T, y.T)

# x = path10[:, 0]
# y = path10[:, 1]
# plt.plot(x.T, y.T)

# plt.xlabel('x/m', fontsize=20)
# plt.ylabel('y/m', fontsize=20)
# plt.show()
####################          test in world two and three       ###########################################