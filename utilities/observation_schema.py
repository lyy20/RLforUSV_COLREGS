"""Shared observation layout for the 3-DOF body-frame policy input.

The layout is versioned because the actuator-state features are part of the
policy contract.  A checkpoint trained with the previous layout must not be
silently resumed with these shifted indices.
"""

OBSERVATION_LAYOUT_VERSION = "3dof_body_frame_v2_action_state"

BASE_FEATURE_NAMES = (
    "surge_u",
    "sway_v",
    "yaw_rate_r",
    "sin_heading",
    "cos_heading",
    "actual_rudder",
    "actual_thrust",
    "previous_smoothed_thrust",
    "previous_smoothed_rudder",
    "target_x_body",
    "target_y_body",
    "target_distance",
    "boundary_clearance",
    "boundary_risk_x_body",
    "boundary_risk_y_body",
)

DYNAMIC_FEATURE_NAMES = (
    "relative_x_body",
    "relative_y_body",
    "distance",
    "size",
    "relative_velocity_x_body",
    "relative_velocity_y_body",
    "type_mask",
)

STATIC_FEATURE_NAMES = (
    "relative_x_body",
    "relative_y_body",
    "distance",
    "size",
    "type_mask",
)

BASE_OBS_DIM = len(BASE_FEATURE_NAMES)
DYNAMIC_OBS_FEATURE_DIM = len(DYNAMIC_FEATURE_NAMES)
STATIC_OBS_FEATURE_DIM = len(STATIC_FEATURE_NAMES)

BASE_INDEX = {name: index for index, name in enumerate(BASE_FEATURE_NAMES)}


def observation_dim(num_dynamic_slots, num_static_slots):
    """Return the exact flat observation width for a configured scene."""
    return (
        BASE_OBS_DIM
        + int(num_dynamic_slots) * DYNAMIC_OBS_FEATURE_DIM
        + int(num_static_slots) * STATIC_OBS_FEATURE_DIM
    )


def dynamic_observation_offset():
    return BASE_OBS_DIM


def static_observation_offset(num_dynamic_slots):
    return BASE_OBS_DIM + int(num_dynamic_slots) * DYNAMIC_OBS_FEATURE_DIM
