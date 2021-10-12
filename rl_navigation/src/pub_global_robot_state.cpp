#include "pub_global_robot_state.h"

std_msgs::Float32MultiArray states;
ros::Publisher  states_publisher;
tf2_ros::Buffer MapBaseLinkTfBuffer;
geometry_msgs::TransformStamped MapBaseLinkTransformStamped;
double roll=0.0, pitch=0.0, yaw=0.0;
geometry_msgs::Quaternion igor_orient;
tf::Quaternion quat;

void odom_callback(const nav_msgs::Odometry::ConstPtr &msg)
{ 
    try
    { 
    MapBaseLinkTransformStamped = MapBaseLinkTfBuffer.lookupTransform("map", "base_footprint" , ros::Time(0));
    }
    catch (tf2::TransformException &ex) 
    {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
    }

    states.data[0] = MapBaseLinkTransformStamped.transform.translation.x;
    states.data[1] = MapBaseLinkTransformStamped.transform.translation.y;
    
    igor_orient = MapBaseLinkTransformStamped.transform.rotation;
    tf::quaternionMsgToTF(igor_orient, quat);
    quat.normalize();
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    states.data[2] = yaw;
    states.data[3] = msg->twist.twist.linear.x; 
    states.data[4] = msg->twist.twist.angular.z; 
    // std::cout<<states.data[0]<<"  "<<states.data[1]<<"  "<<states.data[2]<<"  "<<std::endl;
    states_publisher.publish(states);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "pub_robot_states");
    ros::NodeHandle node;
    
    ros::Subscriber sub_odom = node.subscribe<nav_msgs::Odometry>("/odom",5, odom_callback,ros::TransportHints().tcpNoDelay());
    states_publisher = node.advertise<std_msgs::Float32MultiArray>( "/robot/states", 5);
    
    states.data.resize(5);
    
    tf2_ros::TransformListener MapBaseLinkTfListener(MapBaseLinkTfBuffer);
    ros::Rate rate(20);
    // Wait 2 seconds for the module list to populate, and then print out its contents
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    while (node.ok()){

        ros::spinOnce();
        rate.sleep();
  }
    return 0;
}