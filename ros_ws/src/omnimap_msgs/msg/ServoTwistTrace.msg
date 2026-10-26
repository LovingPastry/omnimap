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
# Frozen-state camera-reference affine rate model, same frame as command.
# Row-major 6x2 matrix: [linear; angular] = B * physical_rates + bias.
# Invalid when linear clipping/slew/deadband modifies the affine command.
bool rate_model_valid
string rate_model_reason
float64 reference_radius
float64[3] reference_scene_center
float64[2] spherical_rates
float64[12] camera_rate_matrix
float64[6] camera_bias
# Limits from this exact Servo cycle, before updating the previous command.
bool linear_limits_valid
float64[3] previous_linear
float64 linear_speed_limit
float64 linear_step_limit
float64 linear_deadband
float64 servo_dt_sec
