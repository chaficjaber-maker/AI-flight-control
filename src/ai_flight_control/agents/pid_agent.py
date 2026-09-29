from dataclasses import dataclass, field

from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.pid import PIDController, PIDGains


@dataclass
class PIDAltitudeAgent:
    """Classical altitude-hold autopilot."""

    gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.006, ki=0.0005, kd=0.15)
    )
    velocity_damping: float = 0.01
    _controller: PIDController | None = None

    def reset(self) -> None:
        if self._controller is None:
            self._controller = PIDController(self.gains)
        else:
            self._controller.reset()

    def act(
        self,
        state: LongitudinalState,
        target_altitude_m: float,
        dt: float,
    ) -> float:
        if self._controller is None:
            self.reset()
        assert self._controller is not None
        error = target_altitude_m - state.altitude_m
        command = self._controller.step(error, dt)
        command -= self.velocity_damping * state.vertical_velocity_m_s
        return max(-1.0, min(1.0, command))
