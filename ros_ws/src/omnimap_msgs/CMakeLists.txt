cmake_minimum_required(VERSION 3.5)
project(omnimap_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  PlannerSnapshotRef.msg
  SphericalCommand.msg
  KinematicStatus.msg
  KinematicConstraint.msg
  KinematicModel.msg
  BridgeStatus.msg
  ServoTwistTrace.msg
)

add_service_files(FILES EvaluateKinematicModel.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
)
