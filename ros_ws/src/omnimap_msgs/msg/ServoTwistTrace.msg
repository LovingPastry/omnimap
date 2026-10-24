# Companion to the unmodified TwistStamped control interface. rospy assigns
# Header.seq itself, so it cannot carry the Planning tick reliably.
# command.header.seq retains the actual outgoing Twist wire sequence (the
# companion is filled after publishing the Twist), alongside its sample stamp.
std_msgs/Header header
uint32 planner_tick
uint32 model_version
time command_stamp
geometry_msgs/TwistStamped command
string reason
