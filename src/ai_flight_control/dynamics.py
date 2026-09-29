from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LongitudinalState:
    """Vertical-channel state for a simplified point-mass aircraft."""

    altitude_m: float
    vertical_velocity_m_s: float


@dataclass(frozen=True)
class LongitudinalParams:
    gravity_m_s2: float = 9.81
    max_thrust_accel_m_s2: float = 25.0
    drag_coeff: float = 0.35


def step_longitudinal(
    state: LongitudinalState,
    normalized_thrust: float,
    dt: float,
    params: LongitudinalParams | None = None,
) -> LongitudinalState:
    """
    Integrate one timestep of a 1-D altitude-hold plant.

    `normalized_thrust` is in [-1, 1] and maps to vertical acceleration
    around hover (0 -> approximately level flight in this model).
    """
    p = params or LongitudinalParams()
    u = float(np.clip(normalized_thrust, -1.0, 1.0))
    thrust_accel = p.gravity_m_s2 + u * (p.max_thrust_accel_m_s2 - p.gravity_m_s2)
    v = state.vertical_velocity_m_s
    accel = thrust_accel - p.gravity_m_s2 - p.drag_coeff * v * abs(v)

    v_next = state.vertical_velocity_m_s + accel * dt
    h_next = state.altitude_m + state.vertical_velocity_m_s * dt + 0.5 * accel * dt * dt

    return LongitudinalState(altitude_m=h_next, vertical_velocity_m_s=v_next)
