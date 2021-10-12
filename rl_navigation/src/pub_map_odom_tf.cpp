#include "pub_odom_tf.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "pub_robot_states");
    ros::NodeHandle node;


    ros::ServiceClient get_state_service;
    gazebo_msgs::GetModelStateResponse objstate;
    gazebo_msgs::GetModelStateRequest model;

    tf2_ros::TransformBroadcaster br1;
    geometry_msgs::TransformStamped transform1;
    tf2_ros::TransformBroadcaster br2;
    geometry_msgs::TransformStamped transform2;

    ros::Time odom_baseLink;
    ros::Time map_odom;

    std_msgs::Float32MultiArray pub_states;
    ros::Publisher  igor_state_publisher;

    geometry_msgs::Quaternion  igor_orient;
    tf::Quaternion quat;

    double sensor_time_last = 0.0;
    double sensor_time_current = 0.0;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;


    get_state_service = node.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    model.model_name = "turtlebot3_burger";

    igor_state_publisher = node.advertise<std_msgs::Float32MultiArray>( "/robot/states", 5);
    ros::Publisher odom_pub = node.advertise<nav_msgs::Odometry>("/odom2", 5);

    nav_msgs::Odometry odom2;
    odom2.header.frame_id = "odom2";

    odom_baseLink = ros::Time::now();
    map_odom = ros::Time::now();

    pub_states.data.resize(5);

    ros::Rate rate(100);

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    while (node.ok()){
        get_state_service.call(model, objstate);
        sensor_time_current = ros::Time::now().toSec();
        map_odom = ros::Time::now();
        transform2.header.stamp = map_odom;

        igor_orient = objstate.pose.orientation;
        tf::quaternionMsgToTF(igor_orient, quat); // the incoming geometry_msgs::Quaternion is transformed to a tf::Quaternion
        quat.normalize(); // normalize the quaternion in case it is not normalized
        //the tf::Quaternion has a method to acess roll pitch and yaw
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        pub_states.data[0] = objstate.pose.position.x;
        pub_states.data[1] = objstate.pose.position.y;
        pub_states.data[2] = yaw;
        pub_states.data[3] = objstate.twist.linear.x;
        pub_states.data[4] = objstate.twist.angular.z;
        igor_state_publisher.publish(pub_states);

        odom2.header.stamp = map_odom;
        //set the position
        odom2.pose.pose.position.x = objstate.pose.position.x;
        odom2.pose.pose.position.y = objstate.pose.position.y;
        odom2.pose.pose.position.z = objstate.pose.position.z;
        odom2.pose.pose.orientation.x = objstate.pose.orientation.x;
        odom2.pose.pose.orientation.y = objstate.pose.orientation.y;
        odom2.pose.pose.orientation.z = objstate.pose.orientation.z;
        odom2.pose.pose.orientation.w = objstate.pose.orientation.w;

        //set the velocity
        odom2.child_frame_id = "base_footprint2";
        odom2.twist.twist.linear.x = objstate.twist.linear.x;
        odom2.twist.twist.linear.y = objstate.twist.linear.y;
        odom2.twist.twist.linear.z = objstate.twist.linear.z;
        odom2.twist.twist.angular.x = objstate.twist.angular.x;
        odom2.twist.twist.angular.y = objstate.twist.angular.y;
        odom2.twist.twist.angular.z = objstate.twist.angular.z;

        //publish the message
        odom_pub.publish(odom2);

        
        transform1.header.stamp = odom_baseLink;
        odom_baseLink = map_odom;
        
        transform1.header.frame_id = "odom2";
        transform1.child_frame_id = "base_footprint2";
        transform1.transform.translation.x = objstate.pose.position.x;
        transform1.transform.translation.y = objstate.pose.position.y;
        transform1.transform.translation.z = objstate.pose.position.z;
        transform1.transform.rotation.x = objstate.pose.orientation.x;
        transform1.transform.rotation.y = objstate.pose.orientation.y;
        transform1.transform.rotation.z = objstate.pose.orientation.z;
        transform1.transform.rotation.w = objstate.pose.orientation.w;
        br1.sendTransform(transform1);
        
        
        transform2.header.frame_id = "map";
        transform2.child_frame_id = "odom2";
        transform2.transform.translation.x = 0;
        transform2.transform.translation.y = 0;
        transform2.transform.translation.z = 0;
        transform2.transform.rotation.x = 0;
        transform2.transform.rotation.y = 0;
        transform2.transform.rotation.z = 0;
        transform2.transform.rotation.w = 1;
        br2.sendTransform(transform2);

        // if (sensor_time_current - sensor_time_last >= 0.01){
        //     sensor_time_last = sensor_time_current;
        //     transform1.header.stamp = odom_baseLink;
        //     odom_baseLink = map_odom;
            
        //     transform1.header.frame_id = "odom2";
        //     transform1.child_frame_id = "base_footprint2";
        //     transform1.transform.translation.x = objstate.pose.position.x;
        //     transform1.transform.translation.y = objstate.pose.position.y;
        //     transform1.transform.translation.z = objstate.pose.position.z;
        //     transform1.transform.rotation.x = objstate.pose.orientation.x;
        //     transform1.transform.rotation.y = objstate.pose.orientation.y;
        //     transform1.transform.rotation.z = objstate.pose.orientation.z;
        //     transform1.transform.rotation.w = objstate.pose.orientation.w;
        //     br1.sendTransform(transform1);
        // }
        ros::spinOnce();
        rate.sleep();
  }
    return 0;
}