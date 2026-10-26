# Read-only model at one aligned joint/TF sample. No command or execution permit.
std_msgs/Header header
time joint_stamp
time status_received_stamp
uint64 model_sequence
string source_frame
string task_frame
string camera_frame
bool valid
string reason
int8 moveit_status
# Tool-origin Jacobian in source_frame; row-major 6x6, model joint order.
float64[36] task_jacobian
# Camera-to-source transform at joint_stamp, row-major 4x4.
float64[16] camera_c2w
float64[3] camera_tool_offset
# Gradient of sigma_min(Dx^-1 J Dq) with respect to joint position.
float64[6] metric_gradient
float64[6] task_velocity_scales
float64[2] rate_scales
float64 sigma_min
float64 derivative_error
bool threshold_configured
float64 m_min
float64 alpha
string[6] joint_names
float64[6] joint_position
float64[6] joint_velocity
float64[6] joint_velocity_lower
float64[6] joint_velocity_upper
bool[6] joint_position_bounded
float64[6] joint_position_lower
float64[6] joint_position_upper
bool[6] joint_acceleration_bounded
float64[6] joint_acceleration_lower
float64[6] joint_acceleration_upper
