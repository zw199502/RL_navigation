cmake_minimum_required(VERSION 2.8.3)
project(rl_navigation)

find_package(catkin REQUIRED roslaunch)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
  tf
  tf2_ros
  sensor_msgs
  geometry_msgs
  tf_conversions
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES igor
  CATKIN_DEPENDS roscpp std_msgs tf tf2_ros sensor_msgs geometry_msgs tf_conversions
#  DEPENDS system_lib
)


#set(EXTRALIB_BIN ${PROJECT_SOURCE_DIR}/my_lib/libiir.so.1.7.4)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${eigen3_INCLUDE_DIRS}
)



add_executable(pub_global_robot_state_node src/pub_global_robot_state.cpp)

target_link_libraries(pub_global_robot_state_node
   ${catkin_LIBRARIES}
 )

add_executable(pub_odom_tf_node src/pub_odom_tf.cpp)

target_link_libraries(pub_odom_tf_node
   ${catkin_LIBRARIES}
 )

add_executable(pub_map_odom_tf_node src/pub_map_odom_tf.cpp)

target_link_libraries(pub_map_odom_tf_node
    ${catkin_LIBRARIES}
  )



catkin_package()
roslaunch_add_file_check(launch)


