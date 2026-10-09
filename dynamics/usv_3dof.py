"""Standalone 3-DOF underactuated USV dynamics model.

The model is intentionally independent from the current multi-agent
environment. It is the future replacement target for the temporary
heading-rate baseline in ``multiagent.core``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Tuple, Union

import numpy as np


ArrayLikeAction = Union["USV3DOFAction", Iterable[float], np.ndarray]


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def body_to_world_vector(psi: float, vector: Iterable[float]) -> np.ndarray:
    """Rotate a planar vector from the USV body frame into the world frame.

    This is the same ``R(psi)`` used by ``eta_dot = R(psi) @ nu``. Body
    ``x`` points forward and body ``y`` points to port (left).
    """
    data = np.asarray(vector, dtype=float).reshape(-1)
    if data.size != 2:
        raise ValueError("A planar body-frame vector must contain exactly two values.")
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))
    return np.array(
        [
            cos_psi * data[0] - sin_psi * data[1],
            sin_psi * data[0] + cos_psi * data[1],
        ],
        dtype=float,
    )


def world_to_body_vector(psi: float, vector: Iterable[float]) -> np.ndarray:
    """Rotate a planar world-frame vector with the inverse ``R(psi).T``."""
    data = np.asarray(vector, dtype=float).reshape(-1)
    if data.size != 2:
        raise ValueError("A planar world-frame vector must contain exactly two values.")
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))
    return np.array(
        [
            cos_psi * data[0] + sin_psi * data[1],
            -sin_psi * data[0] + cos_psi * data[1],
        ],
        dtype=float,
    )


@dataclass
class USV3DOFConfig:
    """Physical and numerical parameters for a 3-DOF USV model."""

    dt: float = 0.1

    # Inertia terms. They can later be replaced by identified vessel data.
    mass_surge: float = 25.0
    mass_sway: float = 35.0
    inertia_yaw: float = 8.0

    # Linear and quadratic damping. Positive values dissipate motion.
    damping_surge: float = 18.0
    damping_surge_quad: float = 6.0
    damping_sway: float = 30.0
    damping_sway_quad: float = 10.0
    damping_yaw: float = 4.0
    damping_yaw_quad: float = 2.0

    # Actuator and hydrodynamic gains.
    rudder_sway_gain: float = 2.0
    rudder_yaw_gain: float = 4.0
    min_rudder_effect_speed: float = 0.2

    # Command limits.
    min_thrust: float = 0.0
    max_thrust: float = 80.0
    max_thrust_rate: float = 120.0
    max_rudder_angle: float = np.deg2rad(35.0)
    max_rudder_rate: float = np.deg2rad(45.0)

    # State limits used to keep simulation numerically sane.
    max_surge_speed: float = 3.0
    max_reverse_speed: float = 0.5
    max_sway_speed: float = 1.0
    max_yaw_rate: float = np.deg2rad(90.0)

    # Constant ocean current in the world frame.
    current_speed: float = 0.0
    current_direction: float = 0.0


@dataclass
class USV3DOFState:
    """USV state in earth-fixed position and body-fixed velocity."""

    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0
    thrust: float = 0.0
    rudder: float = 0.0

    def eta(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi], dtype=float)

    def nu(self) -> np.ndarray:
        return np.array([self.u, self.v, self.r], dtype=float)

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.psi, self.u, self.v, self.r, self.thrust, self.rudder],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "USV3DOFState":
        data = list(values)
        if len(data) != 8:
            raise ValueError("USV3DOFState requires 8 values: x, y, psi, u, v, r, thrust, rudder.")
        return cls(*map(float, data))


@dataclass
class USV3DOFAction:
    """Physical action command before actuator rate limiting."""

    thrust: float
    rudder: float


class USV3DOFModel:
    """3-DOF underactuated USV with thrust and rudder control.

    State convention:
        eta = [x, y, psi] in the earth-fixed frame, with x/y in metres.
        nu = [u, v, r] in the body-fixed frame, with u/v in m/s and r in rad/s.

    Continuous-time structure:
        eta_dot = R(psi) nu + current_world
        M nu_dot = tau(u, rudder, thrust) - D(nu) + coriolis_like_terms

    The equations are intentionally compact. They are suitable as a
    controllable simulation baseline and can later be calibrated with vessel
    identification data.
    """

    def __init__(self, config: USV3DOFConfig | None = None):
        self.config = config or USV3DOFConfig()

    def zero_state(self) -> USV3DOFState:
        return USV3DOFState()

    def _coerce_action_array(self, action: ArrayLikeAction) -> np.ndarray:
        """Convert any supported action representation to a flat float array."""
        if isinstance(action, USV3DOFAction):
            return np.array([action.thrust, action.rudder], dtype=float)
        return np.asarray(action, dtype=float).reshape(-1)

    def normalized_action_to_command(self, action: Iterable[float]) -> USV3DOFAction:
        """Map a future SAC action in [-1, 1]^2 to physical commands."""
        data = self._coerce_action_array(action)
        if data.size != 2:
            raise ValueError("Normalized USV action must contain [thrust, rudder].")
        thrust_scale = _clamp(data[0], -1.0, 1.0)
        rudder_scale = _clamp(data[1], -1.0, 1.0)

        cfg = self.config
        thrust = cfg.min_thrust + (thrust_scale + 1.0) * 0.5 * (cfg.max_thrust - cfg.min_thrust)
        rudder = rudder_scale * cfg.max_rudder_angle
        return USV3DOFAction(thrust=thrust, rudder=rudder)

    def sanitize_physical_action(self, action: ArrayLikeAction) -> USV3DOFAction:
        """Clamp a physical [thrust, rudder] command to actuator bounds."""
        if isinstance(action, USV3DOFAction):
            thrust = action.thrust
            rudder = action.rudder
        else:
            data = self._coerce_action_array(action)
            if data.size != 2:
                raise ValueError("Physical USV action must contain [thrust, rudder].")
            thrust, rudder = data

        cfg = self.config
        return USV3DOFAction(
            thrust=_clamp(thrust, cfg.min_thrust, cfg.max_thrust),
            rudder=_clamp(rudder, -cfg.max_rudder_angle, cfg.max_rudder_angle),
        )

    def step(
        self,
        state: USV3DOFState,
        action: ArrayLikeAction,
        normalized_action: bool = False,
    ) -> USV3DOFState:
        """Advance one step using semi-implicit Euler integration."""
        command = (
            self.normalized_action_to_command(action)
            if normalized_action
            else self.sanitize_physical_action(action)
        )
        actuated = self._apply_actuator_limits(state, command)
        u_dot, v_dot, r_dot = self.body_acceleration(state, actuated)

        cfg = self.config
        next_u = _clamp(
            state.u + u_dot * cfg.dt,
            -cfg.max_reverse_speed,
            cfg.max_surge_speed,
        )
        next_v = _clamp(state.v + v_dot * cfg.dt, -cfg.max_sway_speed, cfg.max_sway_speed)
        next_r = _clamp(state.r + r_dot * cfg.dt, -cfg.max_yaw_rate, cfg.max_yaw_rate)

        x_dot, y_dot, psi_dot = self.earth_fixed_velocity(state.psi, next_u, next_v, next_r)
        return USV3DOFState(
            x=state.x + x_dot * cfg.dt,
            y=state.y + y_dot * cfg.dt,
            psi=wrap_angle(state.psi + psi_dot * cfg.dt),
            u=next_u,
            v=next_v,
            r=next_r,
            thrust=actuated.thrust,
            rudder=actuated.rudder,
        )

    def advance_substeps(
        self,
        state: USV3DOFState,
        action: ArrayLikeAction,
        normalized_action: bool = False,
        substeps: int = 1,
    ) -> USV3DOFState:
        """Advance the dynamics by repeatedly applying one fixed action.

        This keeps the same 0.1 s integration step and actuator limits, but
        avoids repeating action parsing and state-object construction for each
        substep.
        """
        command = (
            self.normalized_action_to_command(action)
            if normalized_action
            else self.sanitize_physical_action(action)
        )

        cfg = self.config
        dt = float(cfg.dt)
        max_thrust_delta = float(cfg.max_thrust_rate * dt)
        max_rudder_delta = float(cfg.max_rudder_rate * dt)
        min_thrust = float(cfg.min_thrust)
        max_thrust = float(cfg.max_thrust)
        min_rudder = float(-cfg.max_rudder_angle)
        max_rudder = float(cfg.max_rudder_angle)
        max_reverse_speed = float(cfg.max_reverse_speed)
        max_surge_speed = float(cfg.max_surge_speed)
        max_sway_speed = float(cfg.max_sway_speed)
        max_yaw_rate = float(cfg.max_yaw_rate)
        mass_surge = float(cfg.mass_surge)
        mass_sway = float(cfg.mass_sway)
        inertia_yaw = float(cfg.inertia_yaw)
        damping_surge = float(cfg.damping_surge)
        damping_surge_quad = float(cfg.damping_surge_quad)
        damping_sway = float(cfg.damping_sway)
        damping_sway_quad = float(cfg.damping_sway_quad)
        damping_yaw = float(cfg.damping_yaw)
        damping_yaw_quad = float(cfg.damping_yaw_quad)
        rudder_sway_gain = float(cfg.rudder_sway_gain)
        rudder_yaw_gain = float(cfg.rudder_yaw_gain)
        min_rudder_effect_speed = float(cfg.min_rudder_effect_speed)
        current_x, current_y = self.current_vector()

        x = float(state.x)
        y = float(state.y)
        psi = float(state.psi)
        u = float(state.u)
        v = float(state.v)
        r = float(state.r)
        thrust_state = float(state.thrust)
        rudder_state = float(state.rudder)

        step_count = max(1, int(round(float(substeps))))
        for _ in range(step_count):
            thrust = _clamp(
                command.thrust,
                thrust_state - max_thrust_delta,
                thrust_state + max_thrust_delta,
            )
            thrust = _clamp(thrust, min_thrust, max_thrust)
            rudder = _clamp(
                command.rudder,
                rudder_state - max_rudder_delta,
                rudder_state + max_rudder_delta,
            )
            rudder = _clamp(rudder, min_rudder, max_rudder)

            rudder_speed = max(abs(u), min_rudder_effect_speed)
            rudder_force = rudder_speed * rudder_speed * np.sin(rudder)

            surge_drag = damping_surge * u + damping_surge_quad * abs(u) * u
            sway_drag = damping_sway * v + damping_sway_quad * abs(v) * v
            yaw_drag = damping_yaw * r + damping_yaw_quad * abs(r) * r

            u_dot = (thrust - surge_drag + mass_sway * v * r) / mass_surge
            v_dot = (rudder_sway_gain * rudder_force - sway_drag - mass_surge * u * r) / mass_sway
            r_dot = (rudder_yaw_gain * rudder_force - yaw_drag) / inertia_yaw

            next_u = _clamp(u + u_dot * dt, -max_reverse_speed, max_surge_speed)
            next_v = _clamp(v + v_dot * dt, -max_sway_speed, max_sway_speed)
            next_r = _clamp(r + r_dot * dt, -max_yaw_rate, max_yaw_rate)

            x_dot = next_u * np.cos(psi) - next_v * np.sin(psi) + current_x
            y_dot = next_u * np.sin(psi) + next_v * np.cos(psi) + current_y

            x += x_dot * dt
            y += y_dot * dt
            psi = wrap_angle(psi + next_r * dt)
            u = next_u
            v = next_v
            r = next_r
            thrust_state = thrust
            rudder_state = rudder

        return USV3DOFState(
            x=x,
            y=y,
            psi=psi,
            u=u,
            v=v,
            r=r,
            thrust=thrust_state,
            rudder=rudder_state,
        )

    def rollout(
        self,
        initial_state: USV3DOFState,
        actions: Iterable[ArrayLikeAction],
        normalized_action: bool = False,
    ) -> Tuple[USV3DOFState, ...]:
        """Return the state trajectory, including the initial state."""
        states = [initial_state]
        state = initial_state
        for action in actions:
            state = self.step(state, action, normalized_action=normalized_action)
            states.append(state)
        return tuple(states)

    def body_acceleration(self, state: USV3DOFState, action: USV3DOFAction) -> Tuple[float, float, float]:
        """Compute body-frame acceleration [u_dot, v_dot, r_dot]."""
        cfg = self.config
        u, v, r = state.u, state.v, state.r

        rudder_speed = max(abs(u), cfg.min_rudder_effect_speed)
        rudder_force = rudder_speed * rudder_speed * np.sin(action.rudder)

        surge_drag = cfg.damping_surge * u + cfg.damping_surge_quad * abs(u) * u
        sway_drag = cfg.damping_sway * v + cfg.damping_sway_quad * abs(v) * v
        yaw_drag = cfg.damping_yaw * r + cfg.damping_yaw_quad * abs(r) * r

        u_dot = (action.thrust - surge_drag + cfg.mass_sway * v * r) / cfg.mass_surge
        v_dot = (cfg.rudder_sway_gain * rudder_force - sway_drag - cfg.mass_surge * u * r) / cfg.mass_sway
        r_dot = (cfg.rudder_yaw_gain * rudder_force - yaw_drag) / cfg.inertia_yaw
        return float(u_dot), float(v_dot), float(r_dot)

    def earth_fixed_velocity(self, psi: float, u: float, v: float, r: float) -> Tuple[float, float, float]:
        """Transform body velocity to earth-fixed velocity and add current."""
        current_x, current_y = self.current_vector()
        hull_velocity = body_to_world_vector(psi, [u, v])
        x_dot = hull_velocity[0] + current_x
        y_dot = hull_velocity[1] + current_y
        return float(x_dot), float(y_dot), float(r)

    def current_vector(self) -> Tuple[float, float]:
        cfg = self.config
        return (
            float(cfg.current_speed * np.cos(cfg.current_direction)),
            float(cfg.current_speed * np.sin(cfg.current_direction)),
        )

    def with_current(self, speed: float, direction: float) -> "USV3DOFModel":
        """Return a copy of the model with updated constant current."""
        return USV3DOFModel(replace(self.config, current_speed=speed, current_direction=direction))

    def _apply_actuator_limits(self, state: USV3DOFState, command: USV3DOFAction) -> USV3DOFAction:
        cfg = self.config
        max_thrust_delta = cfg.max_thrust_rate * cfg.dt
        max_rudder_delta = cfg.max_rudder_rate * cfg.dt

        thrust = _clamp(
            command.thrust,
            state.thrust - max_thrust_delta,
            state.thrust + max_thrust_delta,
        )
        thrust = _clamp(thrust, cfg.min_thrust, cfg.max_thrust)

        rudder = _clamp(
            command.rudder,
            state.rudder - max_rudder_delta,
            state.rudder + max_rudder_delta,
        )
        rudder = _clamp(rudder, -cfg.max_rudder_angle, cfg.max_rudder_angle)
        return USV3DOFAction(thrust=thrust, rudder=rudder)
