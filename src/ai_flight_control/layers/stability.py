from dataclasses import dataclass, field

from ai_flight_control.pid import PIDController, PIDGains
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    EnvelopeLimits,
    FlightState,
    ManeuverSetpoint,
)


@dataclass
class StabilityLayer:
    """
    Stability + critical-parameter function (inner layer).

    AI: adaptive PID gains on altitude-rate / energy tracking.
    Guardian: clamps thrust and vertical-rate requests to envelope limits.
    """

    limits: EnvelopeLimits = field(default_factory=EnvelopeLimits)
    gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.35, ki=0.04, kd=0.12)
    )
    _rate_controller: PIDController | None = None
    _kp_scale: float = 1.0
    _envelope_events: int = 0

    def reset(self) -> None:
        self._rate_controller = PIDController(self.gains)
        self._kp_scale = 1.0
        self._envelope_events = 0

    def adapt(self, context: AdaptationContext) -> None:
        if context.envelope_active:
            self._kp_scale = max(0.6, self._kp_scale - 0.04)
        elif abs(context.vertical_rate_error_m_s) > 1.0:
            self._kp_scale = min(1.5, self._kp_scale + 0.02)

    @property
    def envelope_active_last_step(self) -> bool:
        return self._envelope_events > 0

    def step(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
        if self._rate_controller is None:
            self.reset()

        assert self._rate_controller is not None
        self._envelope_events = 0

        desired_rate = setpoint.target_vertical_rate_m_s
        if state.altitude_m <= self.limits.min_altitude_m and desired_rate < 0:
            desired_rate = 0.0
            self._envelope_events += 1
        if state.altitude_m >= self.limits.max_altitude_m and desired_rate > 0:
            desired_rate = 0.0
            self._envelope_events += 1

        desired_rate = max(
            -self.limits.max_vertical_rate_m_s,
            min(self.limits.max_vertical_rate_m_s, desired_rate),
        )
        if abs(desired_rate) >= self.limits.max_vertical_rate_m_s:
            self._envelope_events += 1

        rate_error = desired_rate - state.vertical_velocity_m_s
        thrust = self._rate_controller.step(rate_error, dt=dt) * self._kp_scale

        if state.vertical_velocity_m_s > self.limits.max_vertical_rate_m_s and thrust > 0:
            thrust = min(thrust, 0.0)
            self._envelope_events += 1
        if state.vertical_velocity_m_s < -self.limits.max_vertical_rate_m_s and thrust < 0:
            thrust = max(thrust, 0.0)
            self._envelope_events += 1

        return ActuatorCommand(normalized_thrust=max(-1.0, min(1.0, thrust)))
