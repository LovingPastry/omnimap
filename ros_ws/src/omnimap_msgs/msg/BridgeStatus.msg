# One record per bridge publication; input stamp survives output re-stamping.
std_msgs/Header header
string camera_frame
string task_frame
string input_topic
string output_topic
geometry_msgs/TwistStamped input
geometry_msgs/TwistStamped output
time status_received_stamp
int8 moveit_status
float64 camera_tf_age
float64 tool_tf_age
bool transform_chain_valid
string reason
