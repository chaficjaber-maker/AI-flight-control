from dataclasses import dataclass, field

import numpy as np

from ai_flight_control.pid import PIDController, PIDGains
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    EnvelopeLimits,
    FlightState,
    ManeuverSetpoint,
    wrap_angle_rad,
)
from ai_flight_control.tecs import AdaptiveTecs


@dataclass
class StabilityLayer:
    """
    Stability + critical-parameter function (inner layer).

    Fixed-wing: adaptive TECS (energy) + bank/heading inner loop with envelope guardian.
    Legacy 1-D: vertical-rate PID when airspeed is not used (see `legacy_vertical_only`).
    """

    limits: EnvelopeLimits = field(default_factory=EnvelopeLimits)
    legacy_vertical_only: bool = False
    gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.35, ki=0.04, kd=0.12)
    )
    bank_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=1.2, ki=0.05, kd=0.25)
    )
    tecs: AdaptiveTecs = field(default_factory=AdaptiveTecs)
    _rate_controller: PIDController | None = None
    _bank_controller: PIDController | None = None
    _kp_scale: float = 1.0
    _bank_scale: float = 1.0
    _envelope_events: int = 0

    def reset(self) -> None:
        self._rate_controller = PIDController(self.gains)
        self._bank_controller = PIDController(self.bank_gains)
        self.tecs.reset()
        self._kp_scale = 1.0
        self._bank_scale = 1.0
        self._envelope_events = 0

    def adapt(self, context: AdaptationContext) -> None:
        if context.envelope_active:
            self._kp_scale = max(0.6, self._kp_scale - 0.04)
            self._bank_scale = max(0.6, self._bank_scale - 0.04)
        elif abs(context.vertical_rate_error_m_s) > 1.0:
            self._kp_scale = min(1.5, self._kp_scale + 0.02)
        if abs(context.heading_error_rad) > np.deg2rad(15.0):
            self._bank_scale = min(1.4, self._bank_scale + 0.02)
        self.tecs.adapt_scales(
            envelope_active=context.envelope_active,
            speed_error=context.airspeed_error_m_s,
        )

    @property
    def envelope_active_last_step(self) -> bool:
        return self._envelope_events > 0

    def step(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
        if self._rate_controller is None or self._bank_controller is None:
            self.reset()

        if self.legacy_vertical_only:
            return self._step_legacy_vertical(state, setpoint, dt)

        return self._step_fixed_wing(state, setpoint, dt)

    def _step_legacy_vertical(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
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

        rate_error = desired_rate - state.vertical_velocity_m_s
        thrust = self._rate_controller.step(rate_error, dt=dt) * self._kp_scale
        return ActuatorCommand.from_legacy_thrust(max(-1.0, min(1.0, thrust)))

    def _step_fixed_wing(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
        assert self._bank_controller is not None
        self._envelope_events = 0

        target_speed = max(
            self.limits.min_airspeed_m_s,
            min(self.limits.max_airspeed_m_s, setpoint.target_airspeed_m_s),
        )
        if state.airspeed_m_s <= self.limits.min_airspeed_m_s * 1.05:
            self._envelope_events += 1

        throttle, pitch = self.tecs.step(
            altitude_m=state.altitude_m,
            airspeed_m_s=state.airspeed_m_s,
            target_altitude_m=setpoint.target_altitude_m,
            target_airspeed_m_s=target_speed,
            dt=dt,
        )

        hdg_err = wrap_angle_rad(setpoint.target_heading_rad - state.heading_rad)
        bank_cmd = self._bank_controller.step(hdg_err, dt=dt) * self._bank_scale
        bank_lim = self.limits.max_bank_rad
        bank_norm = max(-1.0, min(1.0, bank_cmd / bank_lim if bank_lim > 0 else 0.0))

        if abs(setpoint.target_bank_rad) > 0.01:
            bank_norm = max(
                -1.0,
                min(1.0, setpoint.target_bank_rad / bank_lim),
            )

        if abs(bank_norm) >= 0.98:
            self._envelope_events += 1

        if state.altitude_m <= self.limits.min_altitude_m:
            pitch = max(pitch, 0.0)
            self._envelope_events += 1

        return ActuatorCommand(
            throttle=float(max(0.0, min(1.0, throttle))),
            pitch=float(max(-1.0, min(1.0, pitch))),
            roll=float(bank_norm),
        )
