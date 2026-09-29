from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedWingState:
    north_m: float
    east_m: float
    altitude_m: float
    airspeed_m_s: float
    heading_rad: float
    bank_rad: float
    flight_path_angle_rad: float


@dataclass(frozen=True)
class FixedWingParams:
    gravity_m_s2: float = 9.81
    min_airspeed_m_s: float = 12.0
    max_airspeed_m_s: float = 35.0
    max_thrust_accel_m_s2: float = 4.0
    drag_coeff: float = 0.006
    pitch_time_constant_s: float = 0.8
    bank_time_constant_s: float = 0.6
    max_bank_rad: float = np.deg2rad(35.0)
    max_flight_path_angle_rad: float = np.deg2rad(18.0)


@dataclass(frozen=True)
class FixedWingControls:
    """Normalized commands in [-1, 1] unless noted."""

    throttle: float  # maps to [0, 1] internally
    pitch: float  # flight-path angle demand
    roll: float  # bank angle demand


def step_fixed_wing(
    state: FixedWingState,
    controls: FixedWingControls,
    dt: float,
    params: FixedWingParams | None = None,
) -> FixedWingState:
    p = params or FixedWingParams()
    throttle = float(np.clip(controls.throttle, 0.0, 1.0))
    pitch_cmd = float(np.clip(controls.pitch, -1.0, 1.0)) * p.max_flight_path_angle_rad
    bank_cmd = float(np.clip(controls.roll, -1.0, 1.0)) * p.max_bank_rad

    v = max(state.airspeed_m_s, p.min_airspeed_m_s * 0.5)
    gamma = state.flight_path_angle_rad
    bank = state.bank_rad

    gamma_target = pitch_cmd
    bank_target = bank_cmd
    gamma += (gamma_target - gamma) * dt / p.pitch_time_constant_s
    bank += (bank_target - bank) * dt / p.bank_time_constant_s
    gamma = float(np.clip(gamma, -p.max_flight_path_angle_rad, p.max_flight_path_angle_rad))
    bank = float(np.clip(bank, -p.max_bank_rad, p.max_bank_rad))

    thrust_accel = throttle * p.max_thrust_accel_m_s2
    drag_accel = p.drag_coeff * v * v
    v_dot = thrust_accel - drag_accel - p.gravity_m_s2 * np.sin(gamma)
    v_next = float(np.clip(v + v_dot * dt, p.min_airspeed_m_s, p.max_airspeed_m_s))

    psi_dot = p.gravity_m_s2 * np.tan(bank) / max(v_next, p.min_airspeed_m_s)
    h_dot = v_next * np.sin(gamma)
    vn = v_next * np.cos(gamma)

    heading = state.heading_rad + psi_dot * dt
    north = state.north_m + vn * np.cos(heading) * dt
    east = state.east_m + vn * np.sin(heading) * dt
    alt = state.altitude_m + h_dot * dt

    return FixedWingState(
        north_m=north,
        east_m=east,
        altitude_m=alt,
        airspeed_m_s=v_next,
        heading_rad=heading,
        bank_rad=bank,
        flight_path_angle_rad=gamma,
    )
