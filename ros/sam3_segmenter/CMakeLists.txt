cmake_minimum_required(VERSION 3.0.2)
project(sam3_segmenter)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sam3_segmenter_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
