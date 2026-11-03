cmake_minimum_required(VERSION 3.0.2)
project(medical_robotics)

find_package(catkin REQUIRED)

catkin_package()

install(PROGRAMS
  src/view_transition_rl/visual_servoing/controller_rosnode.py
  src/view_transition_rl/visual_servoing/controller_tooltip_rosnode.py
  src/view_transition_rl/visual_servoing/controller_max_rosnode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
