cmake_minimum_required(VERSION 3.0.2)
project(bundlesdf)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  message_filters
  tf2_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/bundlesdf_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
