Title: A Hierarchical Deep Reinforcement Learning Framework with High Efficiency and Generalization for Fast and Safe Navigation
Paper video: https://youtu.be/39LyfzhB47k

1. prerequisites
tensorflow-gpu version >= 2.0
python verison >= 3.5
ubuntu18.04 + ros melodic full desktop which includes Gazebo
Please download https://github.com/osrf/gazebo_models into ~/.gazebo/models

2. Fold turtlebot3_teleop is used for manual contol with keyboard

3. Fold rl_navigation is used for loading Gazebo environments and calling ROS APIs

4. Fold move_base is slightly revised from the ros-melodic-navigation package.

5. Fold RL include DRL training and testing codes and results.

6. create ROS project

(1) mkdir ws_RL

(2) cd ws_RL

(3) mkdir src

(4) copy the above four folds into src

(5) cd ws_RL

(6) catkin_make

please install corresponding ROS packages when errors occur

7. execute training

(1) cd ws_RL

(2) source devel/setup.bash

(3) roslaunch rl_navigation roslaunch rl_navigation turtlebot3_stage_1.launch

(4) cd ws_RL/src/RL

I.   running baseline one with DDPG    python DDPG.py

II.  running baseline two with DDQN    python DDQN.py

III. running pure DRL python comparison_normal.py

I and II are baselines

III. running the low-level DRL         python navigation_DQN_5.py

IV.  after running the low-level DRL, 

     then running the high-level DRL   python navigation_DQN_5_goal.py

III and IV are applied for simulation

V.   if running the low-level DRL 

     for the igor robot                python navigation_DQN_igor.py   

VI.  if running the high-level DRL 

     for the igor robot                python navigation_DQN_igor_goal.py   

V and VI are only used for real implementations.

8. the training log including reward and neural network weights is in the fold 'model'

9. execute testing

There are three testing environments. If you want to change testing environments,

please revise the code in file rl_navigation turtlebot3_stage_1.launch.

The default code is <arg name="world_name" value="$(find rl_navigation)/worlds/train_world1.world"/>.

If you want to use world2, 

then the code should be <arg name="world_name" value="$(find rl_navigation)/worlds/train_world2.world"/>.

If you want to use world3, 

then the code should be <arg name="world_name" value="$(find rl_navigation)/worlds/train_world3.world"/>.

Correspondingly, you need revise the code in global_planner2.launch

The default code is <arg name="map_file" default="$(find rl_navigation)/maps/map.yaml"/>.

If you want to use world2, 

then the code should be <arg name="map_file" default="$(find rl_navigation)/maps/map2.yaml"/>.

If you want to use world3, 

then the code should be <arg name="map_file" default="$(find rl_navigation)/maps/map3.yaml"/>.

Then, you can run testing code as below:

(1) cd ws_RL

(2) source devel/setup.bash

(3) roslaunch rl_navigation roslaunch rl_navigation turtlebot3_stage_1.launch

(4) cd ws_RL/src/RL

I.   running baseline one with DDPG    python DDPG_test.py

II.  running baseline two with DDQN    python DDQN_test.py

I and II are baselines

III. running the low-level DRL         python navigation_DQN_5_test.py

IV.  running the high-level DRL        python navigation_DQN_5_goal_test.py

III and IV are applied for simulation

V.   runing dwa(move_base)

firstly, roslaunch rl_navigation turtlebot3_navigation.launch

then python dwa_test.py

10. the testing log is in the fold data_processing.

11. data processing

after obtaining testing data, you can plot them by running python plot_map_path.py

12. you can also build map mannually by running 

roslaunch rl_navigation turtlebot3_gmapping.launch
