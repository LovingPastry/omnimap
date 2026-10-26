# Advisory local model only. Includes RobotModel joint velocity bounds, but not
# acceleration, camera speed, collision or joint-position constraints. Physical spherical
# rates = rate_scales * u; a dot u >= c uses dimensionless u.
std_msgs/Header header
time joint_stamp
time twist_stamp
uint32 twist_seq
time command_stamp
uint32 planner_tick
uint32 model_version
string source_frame
string task_frame
bool valid
string reason
float64[2] rate_scales
float64 reference_radius
float64[3] reference_scene_center
float64[2] linearization_rates
float64[2] a
float64 drift
float64 sigma_min
float64 inverse_residual
float64 derivative_error
# c is usable only with a finite, explicitly configured m_min and alpha.
bool threshold_configured
float64 m_min
float64 alpha
float64 c
# qdot = joint_rate_matrix * u + joint_rate_bias, rad/s, row-major 6x2.
# All rows follow joint_names, exactly the monitor's MoveIt group order.
string[6] joint_names
float64[12] joint_rate_matrix
float64[6] joint_rate_bias
float64[6] joint_velocity_lower
float64[6] joint_velocity_upper
