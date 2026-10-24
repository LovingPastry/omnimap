// Read-only monitor. Never publishes a control command or infers a Jacobian
// from MoveIt status. Uses the same RobotModel and group as Servo.
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Int8.h>
#include <omnimap_msgs/BridgeStatus.h>
#include <omnimap_msgs/KinematicStatus.h>
#include <omnimap_msgs/ServoTwistTrace.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/SVD>
#include <deque>
#include <set>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>

namespace
{
const char* statusName(int code)
{
  switch (code)
  {
    case 0: return "NO_WARNING";
    case 1: return "DECELERATE_FOR_SINGULARITY";
    case 2: return "HALT_FOR_SINGULARITY";
    case 3: return "DECELERATE_FOR_COLLISION";
    case 4: return "HALT_FOR_COLLISION";
    case 5: return "JOINT_BOUND";
    default: return "UNKNOWN";
  }
}

Eigen::Matrix<double, 6, 1> twist(const geometry_msgs::TwistStamped& msg)
{
  Eigen::Matrix<double, 6, 1> result;
  result << msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z;
  return result;
}

std::string frame(std::string name)
{
  while (!name.empty() && name.front() == '/') name.erase(0, 1);
  return name;
}

Eigen::Isometry3d pose(const geometry_msgs::Transform& msg)
{
  Eigen::Quaterniond q(msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z);
  if (!q.coeffs().allFinite() || std::abs(q.norm()-1.0) > 1e-3)
    throw std::runtime_error("tf_invalid_rotation");
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = q.normalized().toRotationMatrix();
  result.translation() << msg.translation.x, msg.translation.y, msg.translation.z;
  if (!result.matrix().allFinite()) throw std::runtime_error("tf_nonfinite");
  return result;
}
}

class KinematicMonitor
{
public:
  KinematicMonitor() : pnh_("~"), tf_listener_(tf_)
  {
    pnh_.param<std::string>("servo_ns", servo_ns_, "/servo_server");
    pnh_.param<std::string>("robot_description", description_, "robot_description");
    ros::NodeHandle servo(servo_ns_);
    std::string group, joint_topic, status_topic, command_frame;
    // Fail at startup instead of silently monitoring a different group/frame.
    if (!servo.getParam("move_group_name", group) || !servo.getParam("planning_frame", base_) ||
        !servo.getParam("ee_frame_name", tool_) || !servo.getParam("joint_topic", joint_topic) ||
        !servo.getParam("status_topic", status_topic) ||
        !servo.getParam("robot_link_command_frame", command_frame) ||
        !servo.getParam("cartesian_command_in_topic", output_topic_))
      throw std::runtime_error("missing required Servo parameters under " + servo_ns_);
    base_ = frame(base_); tool_ = frame(tool_);
    if (frame(command_frame) != base_)
      throw std::runtime_error("requires Servo robot_link_command_frame == planning_frame");
    joint_topic = servo.resolveName(joint_topic);
    status_topic = servo.resolveName(status_topic);
    output_topic_ = servo.resolveName(output_topic_);
    pnh_.param<std::string>("camera_frame", camera_, "cam_1_color_optical_frame");
    camera_ = frame(camera_);
    if (base_.empty() || tool_.empty() || camera_.empty()) throw std::runtime_error("empty frame");
    timeout_ = positive("sample_timeout", 0.3);
    skew_ = positive("max_sample_skew", 0.1);
    position_tolerance_ = positive("tf_position_tolerance", 0.01);
    angle_tolerance_ = positive("tf_angle_tolerance", 0.03);
    const double hz = positive("publish_hz", 20.0);
    robot_model_loader::RobotModelLoader::Options options(description_);
    options.load_kinematics_solvers_ = false;
    loader_.reset(new robot_model_loader::RobotModelLoader(options));
    model_ = loader_->getModel();
    if (!model_) throw std::runtime_error("unable to load " + description_);
    group_ = model_->getJointModelGroup(group);
    if (!group_ || !group_->isChain() || group_->getVariableCount() != 6 ||
        group_->getLinkModelNames().back() != tool_)
      throw std::runtime_error("requires a six-variable chain ending at Servo ee_frame_name");
    for (const auto& name : {base_, tool_})
      if (!model_->hasLinkModel(name)) throw std::runtime_error("RobotModel missing link " + name);
    state_.reset(new moveit::core::RobotState(model_));
    state_->setToDefaultValues();
    state_->update();
    names_ = group_->getVariableNames();
    required_names_.insert(names_.begin(), names_.end());
    // Moving ancestors must never be left at RobotState defaults.
    for (const auto* link : {model_->getLinkModel(base_), model_->getLinkModel(tool_)})
      for (auto* joint = link->getParentJointModel(); joint; )
      {
        const auto& vars = joint->getVariableNames();
        required_names_.insert(vars.begin(), vars.end());
        auto* parent = joint->getParentLinkModel();
        joint = parent ? parent->getParentJointModel() : nullptr;
      }
    std::vector<double> dx;
    if (!pnh_.getParam("task_velocity_scales", dx) || dx.size() != 6)
      throw std::runtime_error("task_velocity_scales must contain six positive values");
    for (size_t i = 0; i < 6; ++i)
    {
      const auto& bound = model_->getVariableBounds(names_[i]);
      if (!bound.velocity_bounded_ || !std::isfinite(bound.max_velocity_) || bound.max_velocity_ <= 0 ||
          !std::isfinite(dx[i]) || dx[i] <= 0)
        throw std::runtime_error("invalid task scale or RobotModel velocity bound for " + names_[i]);
      dx_[i] = dx[i]; dq_[i] = bound.max_velocity_;
    }
    std::ostringstream ordering;
    for (const auto& name : names_) ordering << name << ' ';
    ROS_INFO_STREAM("kinematic_model: description=" << description_ << " model_frame=" << model_->getModelFrame()
                    << " group=" << group << " source=" << base_ << " task=" << tool_
                    << " order=[" << ordering.str() << "] Dx=[" << dx_.transpose()
                    << "] Dq=[" << dq_.transpose() << "]");
    std::string topic;
    pnh_.param<std::string>("kinematic_status_topic", topic, "/omnimap/kinematic_status");
    pub_ = nh_.advertise<omnimap_msgs::KinematicStatus>(topic, 10);
    joints_sub_ = nh_.subscribe(joint_topic, 1, &KinematicMonitor::joints, this);
    status_sub_ = nh_.subscribe(status_topic, 1, &KinematicMonitor::status, this);
    pnh_.param<std::string>("bridge_status_topic", topic, "/omnimap/bridge_status");
    bridge_sub_ = nh_.subscribe(topic, 10, &KinematicMonitor::bridge, this);
    pnh_.param<std::string>("twist_trace_topic", topic, "/omnimap/servo_twist_trace");
    trace_sub_ = nh_.subscribe(topic, 100, &KinematicMonitor::trace, this);
    timer_ = nh_.createWallTimer(ros::WallDuration(1.0/hz), &KinematicMonitor::publish, this);
  }

private:
  double positive(const std::string& name, double value)
  {
    pnh_.param(name, value, value);
    if (!std::isfinite(value) || value <= 0) throw std::runtime_error(name + " must be positive and finite");
    return value;
  }

  void joints(const sensor_msgs::JointStateConstPtr& msg)
  {
    joints_ = msg; joints_received_ = ros::SteadyTime::now();
    joints_error_.clear();
    if (msg->name.size() != msg->position.size() || msg->name.size() != msg->velocity.size())
      joints_error_ = "joint_array_size_mismatch";
    else
    {
      std::set<std::string> seen;
      for (size_t i = 0; i < msg->name.size(); ++i)
      {
        if (!seen.insert(msg->name[i]).second || !std::isfinite(msg->position[i]) || !std::isfinite(msg->velocity[i]))
          joints_error_ = "joint_duplicate_or_nonfinite";
      }
      for (const auto& name : required_names_)
        if (!seen.count(name)) joints_error_ = "joint_missing:" + name;
      if (joints_error_.empty())
      {
        // Assign by name, never by the incoming /joint_states array order.
        for (size_t i = 0; i < msg->name.size(); ++i)
          if (std::find(model_->getVariableNames().begin(), model_->getVariableNames().end(), msg->name[i]) !=
              model_->getVariableNames().end())
          {
            state_->setVariablePosition(msg->name[i], msg->position[i]);
            state_->setVariableVelocity(msg->name[i], msg->velocity[i]);
          }
        state_->update();
      }
    }
  }

  void status(const std_msgs::Int8ConstPtr& msg)
  {
    status_ = msg->data; status_stamp_ = ros::Time::now(); status_received_ = ros::SteadyTime::now();
  }
  void bridge(const omnimap_msgs::BridgeStatusConstPtr& msg)
  {
    bridge_ = msg; bridge_received_ = ros::SteadyTime::now();
  }
  void trace(const omnimap_msgs::ServoTwistTraceConstPtr& msg)
  {
    traces_.push_back(msg);
    if (traces_.size() > 200) traces_.pop_front();
  }

  void fresh(const ros::Time& stamp, const ros::SteadyTime& received, const std::string& label) const
  {
    const double age = (ros::Time::now()-stamp).toSec();
    if (stamp.isZero() || age < -0.05 || age > timeout_ ||
        (ros::SteadyTime::now()-received).toSec() > timeout_)
      throw std::runtime_error(label + "_stale_or_invalid_stamp");
  }

  void checkTF(const std::string& link)
  {
    const auto transform = tf_.lookupTransform(base_, link, joints_->header.stamp, ros::Duration(0));
    const auto actual = pose(transform.transform);
    // The RealSense optical frame can be supplied by the calibrated hand-eye
    // TF chain rather than embedded in robot_description. Only tool0 is the
    // Jacobian reference link; never invent a second camera model here.
    if (!model_->hasLinkModel(link)) return;
    const Eigen::Isometry3d expected = state_->getGlobalLinkTransform(base_).inverse()*state_->getGlobalLinkTransform(link);
    if ((actual.translation()-expected.translation()).norm() > position_tolerance_ ||
        Eigen::AngleAxisd(expected.linear().transpose()*actual.linear()).angle() > angle_tolerance_)
      throw std::runtime_error("tf_robot_model_mismatch:" + link);
  }

  void publish(const ros::WallTimerEvent&)
  {
    omnimap_msgs::KinematicStatus out;
    out.header.stamp = ros::Time::now(); out.header.frame_id = base_;
    out.source_frame = base_; out.task_frame = tool_;
    out.moveit_status = status_; out.moveit_status_text = statusName(status_);
    out.sigma_min = out.sigma_max = out.condition_number = out.manipulability =
        out.command_speed_norm = out.actual_speed_norm = std::numeric_limits<double>::quiet_NaN();
    double raw_condition = std::numeric_limits<double>::quiet_NaN();
    ros::Time command_stamp;
    try
    {
      if (!joints_) throw std::runtime_error("joint_states_missing");
      fresh(joints_->header.stamp, joints_received_, "joint_states");
      if (!joints_error_.empty()) throw std::runtime_error(joints_error_);
      fresh(status_stamp_, status_received_, "moveit_status");
      // MoveIt getJacobian expresses columns in the group-root parent frame.
      // Rotate BOTH blocks to Servo planning_frame; reference point is tool0's
      // origin already, so there is no extra reference-point translation here.
      Eigen::MatrixXd jacobian;
      if (!state_->getJacobian(group_, model_->getLinkModel(tool_), Eigen::Vector3d::Zero(), jacobian))
        throw std::runtime_error("getJacobian_failed");
      const auto* parent = group_->getJointModels().front()->getParentLinkModel();
      Eigen::Matrix3d parent_rotation = Eigen::Matrix3d::Identity();
      if (parent) parent_rotation = state_->getGlobalLinkTransform(parent).linear();
      const Eigen::Matrix3d rotation = state_->getGlobalLinkTransform(base_).linear().transpose()*parent_rotation;
      jacobian.topRows(3) = (rotation*jacobian.topRows(3)).eval();
      jacobian.bottomRows(3) = (rotation*jacobian.bottomRows(3)).eval();
      if (!jacobian.allFinite()) throw std::runtime_error("jacobian_nonfinite");
      const Eigen::MatrixXd normalized = dx_.cwiseInverse().asDiagonal()*jacobian*dq_.asDiagonal();
      if (!normalized.allFinite()) throw std::runtime_error("normalized_jacobian_nonfinite");
      const Eigen::VectorXd values = Eigen::JacobiSVD<Eigen::MatrixXd>(normalized).singularValues();
      out.sigma_min = values[values.size()-1]; out.sigma_max = values[0];
      if (out.sigma_min <= 1e-12) out.sigma_min = 0;
      out.condition_number = out.sigma_min == 0 ? std::numeric_limits<double>::infinity() : out.sigma_max/out.sigma_min;
      out.manipulability = values.prod();
      if (!std::isfinite(out.sigma_max) || !std::isfinite(out.manipulability))
        throw std::runtime_error("kinematic_metrics_nonfinite");
      const Eigen::VectorXd raw = Eigen::JacobiSVD<Eigen::MatrixXd>(jacobian).singularValues();
      raw_condition = raw[raw.size()-1] <= 1e-12 ? std::numeric_limits<double>::infinity() : raw[0]/raw[raw.size()-1];
      Eigen::Matrix<double, 6, 1> velocity;
      for (size_t i = 0; i < 6; ++i) velocity[i] = state_->getVariableVelocity(names_[i]);
      out.actual_speed_norm = (dx_.cwiseInverse().asDiagonal()*jacobian*velocity).norm();
      if (!std::isfinite(out.actual_speed_norm)) throw std::runtime_error("actual_speed_nonfinite");
      checkTF(tool_); checkTF(camera_);
      if (!bridge_) throw std::runtime_error("bridge_missing");
      fresh(bridge_->header.stamp, bridge_received_, "bridge");
      // Source freshness must survive a bridge that republishes old data.
      fresh(bridge_->input.header.stamp, bridge_received_, "bridge_input");
      if (!bridge_->transform_chain_valid) throw std::runtime_error("bridge_invalid:" + bridge_->reason);
      if (frame(bridge_->input.header.frame_id) != base_ || frame(bridge_->output.header.frame_id) != base_)
        throw std::runtime_error("bridge_frame_mismatch");
      if (frame(bridge_->camera_frame) != camera_ || frame(bridge_->task_frame) != tool_ ||
          bridge_->output_topic != output_topic_)
        throw std::runtime_error("bridge_reference_or_topic_mismatch");
      if (std::abs((joints_->header.stamp-bridge_->input.header.stamp).toSec()) > skew_)
        throw std::runtime_error("joint_command_sample_skew");
      if (bridge_->moveit_status != status_ || bridge_->status_received_stamp.isZero() ||
          std::abs((status_stamp_-bridge_->status_received_stamp).toSec()) > skew_)
        throw std::runtime_error("bridge_status_sample_skew");
      const auto command = twist(bridge_->output);
      if (!command.allFinite()) throw std::runtime_error("bridge_output_nonfinite");
      out.command_speed_norm = (dx_.cwiseInverse().asDiagonal()*command).norm();
      if (!std::isfinite(out.command_speed_norm)) throw std::runtime_error("command_speed_nonfinite");
      for (auto it = traces_.rbegin(); it != traces_.rend(); ++it)
      {
        const auto& t = **it;
        if (t.command.header.stamp == bridge_->input.header.stamp &&
            t.command.header.seq == bridge_->input.header.seq &&
            frame(t.command.header.frame_id) == base_ &&
            (twist(t.command)-twist(bridge_->input)).squaredNorm() == 0)
        {
          out.planner_tick = t.planner_tick; command_stamp = t.command_stamp;
          break;
        }
      }
      out.transform_chain_valid = true;
      out.reason = command_stamp.isZero() ? "ok:tick_unavailable" : "ok";
    }
    catch (const std::exception& exc) { out.reason = exc.what(); }
    pub_.publish(out);
    ROS_INFO_STREAM("kinematic_sample: stamp=" << out.header.stamp << " tick=" << out.planner_tick
                    << " command_stamp=" << command_stamp
                    << " joint_stamp=" << (joints_ ? joints_->header.stamp : ros::Time())
                    << " input_stamp=" << (bridge_ ? bridge_->input.header.stamp : ros::Time())
                    << " input_seq=" << (bridge_ ? bridge_->input.header.seq : 0)
                    << " output_stamp=" << (bridge_ ? bridge_->output.header.stamp : ros::Time())
                    << " output_seq=" << (bridge_ ? bridge_->output.header.seq : 0)
                    << " status_stamp=" << status_stamp_ << " status=" << status_ << ':' << statusName(status_)
                    << " sigma_min=" << out.sigma_min << " sigma_max=" << out.sigma_max
                    << " condition=" << out.condition_number << " raw_condition=" << raw_condition
                    << " manipulability=" << out.manipulability << " command_norm=" << out.command_speed_norm
                    << " actual_norm=" << out.actual_speed_norm << " valid=" << bool(out.transform_chain_valid)
                    << " reason=" << out.reason);
  }

  ros::NodeHandle nh_, pnh_;
  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<robot_model_loader::RobotModelLoader> loader_;
  moveit::core::RobotModelPtr model_;
  std::unique_ptr<moveit::core::RobotState> state_;
  const moveit::core::JointModelGroup* group_ = nullptr;
  std::vector<std::string> names_;
  std::set<std::string> required_names_;
  Eigen::Matrix<double, 6, 1> dx_, dq_;
  std::string servo_ns_, description_, base_, tool_, camera_, joints_error_, output_topic_;
  double timeout_, skew_, position_tolerance_, angle_tolerance_;
  int status_ = -1;
  sensor_msgs::JointStateConstPtr joints_;
  omnimap_msgs::BridgeStatusConstPtr bridge_;
  std::deque<omnimap_msgs::ServoTwistTraceConstPtr> traces_;
  ros::SteadyTime joints_received_, status_received_, bridge_received_;
  ros::Time status_stamp_;
  ros::Subscriber joints_sub_, status_sub_, bridge_sub_, trace_sub_;
  ros::Publisher pub_;
  ros::WallTimer timer_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "omnimap_kinematic_monitor");
  try { KinematicMonitor monitor; ros::spin(); }
  catch (const std::exception& exc) { ROS_FATAL_STREAM(exc.what()); return 1; }
  return 0;
}
