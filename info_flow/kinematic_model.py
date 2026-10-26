"""Build advisory rate constraints from one authoritative MoveIt model sample."""
import math
import copy
import numpy as np


def constraint_from_model(model, trace, *, received_wall, ros_now, wall_now, timeout_sec=.3):
    """Use a camera affine candidate computed at model.joint_stamp.

    No new RobotModel is loaded here. The producer owns J, metric gradient,
    frame alignment and model bounds. This conversion does not authorize
    execution or claim that the sampled model is the future physical state.
    """
    from omnimap_msgs.msg import KinematicConstraint
    if not model.valid or model.moveit_status not in (0, 1):
        raise ValueError("snapshot invalid:"+model.reason)
    if (not math.isfinite(timeout_sec) or timeout_sec <= 0
            or not all(math.isfinite(x) for x in (received_wall, ros_now, wall_now))
            or not 0 <= wall_now-received_wall <= timeout_sec):
        raise ValueError("snapshot receipt stale")
    for stamp in (model.header.stamp, model.joint_stamp, model.status_received_stamp, trace.command_stamp):
        age = ros_now-stamp.to_sec()
        if stamp.is_zero() or not math.isfinite(age) or not -.05 <= age <= timeout_sec:
            raise ValueError("snapshot source stale")
    source = model.source_frame.lstrip("/")
    if (not source or model.header.frame_id.lstrip("/") != source or not model.task_frame
            or not model.camera_frame or trace.command.header.frame_id.lstrip("/") != source
            or trace.command.header.stamp != model.joint_stamp or trace.reason != "preview"
            or not trace.rate_model_valid or not trace.linear_limits_valid):
        raise ValueError("snapshot/candidate provenance mismatch")
    def array(value, shape):
        value = np.asarray(value, dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError("snapshot/candidate array invalid")
        return value
    if (model.model_sequence <= 0 or len(model.joint_names) != 6
            or len(set(model.joint_names)) != 6 or any(not n.strip() for n in model.joint_names)):
        raise ValueError("snapshot joint identity invalid")
    position = array(model.joint_position, (6,))
    array(model.joint_velocity, (6,))
    lower, upper = array(model.joint_velocity_lower, (6,)), array(model.joint_velocity_upper, (6,))
    p_lo, p_hi = array(model.joint_position_lower, (6,)), array(model.joint_position_upper, (6,))
    a_lo, a_hi = array(model.joint_acceleration_lower, (6,)), array(model.joint_acceleration_upper, (6,))
    p_mask, a_mask = np.asarray(model.joint_position_bounded), np.asarray(model.joint_acceleration_bounded)
    if p_mask.shape != (6,) or a_mask.shape != (6,):
        raise ValueError("snapshot bound flags invalid")
    p_mask, a_mask = p_mask.astype(bool), a_mask.astype(bool)
    if (np.any(lower >= 0) or np.any(upper <= 0) or np.any(p_lo[p_mask] >= p_hi[p_mask])
            or np.any(position[p_mask] < p_lo[p_mask]) or np.any(position[p_mask] > p_hi[p_mask])
            or np.any(a_lo[a_mask] >= 0) or np.any(a_hi[a_mask] <= 0)):
        raise ValueError("snapshot bounds invalid")
    array(trace.reference_scene_center, (3,))
    array(trace.previous_linear, (3,))
    limits = (trace.linear_speed_limit, trace.linear_step_limit, trace.linear_deadband)
    if (not all(math.isfinite(x) and x >= 0 for x in limits)
            or not math.isfinite(trace.reference_radius) or trace.reference_radius <= 0
            or not math.isfinite(trace.servo_dt_sec) or trace.servo_dt_sec <= 0):
        raise ValueError("snapshot candidate limits invalid")
    jac = array(model.task_jacobian, (36,)).reshape(6, 6)
    pose = array(model.camera_c2w, (16,)).reshape(4, 4)
    gradient = array(model.metric_gradient, (6,))
    dx, scales = array(model.task_velocity_scales, (6,)), array(model.rate_scales, (2,))
    dq = array(model.joint_velocity_upper, (6,))
    if (np.any(dx <= 0) or np.any(scales <= 0) or np.any(dq <= 0)
            or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
            or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., rtol=0, atol=1e-6)):
        raise ValueError("snapshot scales or pose invalid")
    sigma = np.linalg.svd(jac/dx[:, None]*dq[None, :], compute_uv=False)
    if (not math.isfinite(model.sigma_min) or not np.isclose(sigma[-1], model.sigma_min, rtol=1e-6, atol=1e-9)
            or sigma[-1] <= 1e-6 or sigma[-2]-sigma[-1] <= 1e-5*max(1., sigma[0])
            or not math.isfinite(model.derivative_error) or model.derivative_error < 0
            or not math.isfinite(model.alpha) or model.alpha <= 0):
        raise ValueError("snapshot metric invalid")
    camera = array(trace.camera_rate_matrix, (12,)).reshape(6, 2)
    bias = array(trace.camera_bias, (6,))
    rates = array(trace.spherical_rates, (2,))
    command = trace.command.twist
    actual = np.array([command.linear.x, command.linear.y, command.linear.z,
                       command.angular.x, command.angular.y, command.angular.z])
    if not np.allclose(camera@rates+bias, actual, rtol=0, atol=1e-9):
        raise ValueError("snapshot candidate reconstruction mismatch")
    offset = array(model.camera_tool_offset, (3,))
    task, drift_task = camera.copy(), bias.copy()
    for col in range(2):
        task[:3, col] -= np.cross(camera[3:, col], offset)
    drift_task[:3] -= np.cross(bias[3:], offset)
    requested = np.column_stack((task*scales, drift_task))
    singular = np.linalg.svd(jac, compute_uv=False)
    if singular[-1] <= singular[0]*1e-8:
        raise ValueError("snapshot inverse rank deficient")
    directions = np.linalg.solve(jac, requested)
    residual = np.linalg.norm((jac@directions-requested)/dx[:, None])
    if not math.isfinite(residual) or residual > 1e-6:
        raise ValueError("snapshot inverse residual")
    out = KinematicConstraint()
    out.header = copy.deepcopy(model.header)
    out.source_frame, out.task_frame = model.source_frame, model.task_frame
    out.joint_stamp = copy.deepcopy(model.joint_stamp)
    out.twist_stamp = copy.deepcopy(model.joint_stamp)
    out.twist_seq = trace.command.header.seq
    out.command_stamp, out.planner_tick, out.model_version = copy.deepcopy(trace.command_stamp), trace.planner_tick, trace.model_version
    out.rate_scales = scales.tolist()
    out.reference_radius, out.reference_scene_center = trace.reference_radius, list(trace.reference_scene_center)
    out.linearization_rates = list(trace.spherical_rates)
    out.a = (gradient@directions[:, :2]).tolist()
    out.drift = float(gradient@directions[:, 2])
    out.sigma_min, out.derivative_error, out.inverse_residual = model.sigma_min, model.derivative_error, residual
    out.threshold_configured, out.m_min, out.alpha = model.threshold_configured, model.m_min, model.alpha
    if out.threshold_configured and (not math.isfinite(out.m_min) or out.m_min <= 0):
        raise ValueError("snapshot threshold invalid")
    out.c = -out.alpha*(out.sigma_min-out.m_min)-out.drift if out.threshold_configured else float("nan")
    out.joint_rate_matrix, out.joint_rate_bias = directions[:, :2].reshape(-1).tolist(), directions[:, 2].tolist()
    for name in ("joint_names", "joint_position", "joint_velocity", "joint_velocity_lower", "joint_velocity_upper",
                 "joint_position_bounded", "joint_position_lower", "joint_position_upper",
                 "joint_acceleration_bounded", "joint_acceleration_lower", "joint_acceleration_upper"):
        setattr(out, name, list(getattr(model, name)))
    out.camera_linear_matrix, out.camera_linear_bias = (camera[:3]*scales).reshape(-1).tolist(), bias[:3].tolist()
    out.previous_linear = list(trace.previous_linear)
    out.linear_speed_limit, out.linear_step_limit = trace.linear_speed_limit, trace.linear_step_limit
    out.linear_deadband, out.servo_dt_sec = trace.linear_deadband, trace.servo_dt_sec
    out.valid, out.reason = True, "advisory_preview:model_snapshot"
    return out
