from dataclasses import dataclass

from ai_flight_control.agents.base import FlightAgent
from ai_flight_control.dynamics import LongitudinalParams, LongitudinalState, step_longitudinal


@dataclass
class SimulationResult:
    times_s: list[float]
    altitudes_m: list[float]
    vertical_velocities_m_s: list[float]
    commands: list[float]
    target_altitude_m: float


def run_altitude_hold(
    agent: FlightAgent,
    *,
    initial: LongitudinalState,
    target_altitude_m: float,
    duration_s: float,
    dt: float,
    params: LongitudinalParams | None = None,
) -> SimulationResult:
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration_s and dt must be positive")

    agent.reset()
    state = initial
    times: list[float] = []
    altitudes: list[float] = []
    velocities: list[float] = []
    commands: list[float] = []

    t = 0.0
    steps = int(duration_s / dt)
    for _ in range(steps + 1):
        times.append(t)
        altitudes.append(state.altitude_m)
        velocities.append(state.vertical_velocity_m_s)

        cmd = agent.act(state, target_altitude_m, dt)
        commands.append(cmd)
        if t >= duration_s:
            break
        state = step_longitudinal(state, cmd, dt, params)
        t += dt

    return SimulationResult(
        times_s=times,
        altitudes_m=altitudes,
        vertical_velocities_m_s=velocities,
        commands=commands,
        target_altitude_m=target_altitude_m,
    )
