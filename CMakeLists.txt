cmake_minimum_required(VERSION 3.0.2)
project(bundlesdf)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  tf
  tf2_ros
)

catkin_python_setup()

catkin_package()

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/bundlesdf_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
