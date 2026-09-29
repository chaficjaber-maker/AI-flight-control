from dataclasses import dataclass, field

from ai_flight_control.pid import PIDController, PIDGains


@dataclass
class TecsGains:
    energy_throttle: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.08, ki=0.015, kd=0.02)
    )
    balance_pitch: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.012, ki=0.002, kd=0.04)
    )


@dataclass
class AdaptiveTecs:
    """
    Simplified total-energy style longitudinal controller.

    Throttle drives combined height + speed error; pitch demand trades energy
    between altitude and airspeed. Gains scale online via `adapt_scales`.
    """

    cruise_airspeed_m_s: float = 22.0
    gains: TecsGains = field(default_factory=TecsGains)
    speed_weight: float = 1.0
    _throttle_pid: PIDController | None = None
    _pitch_pid: PIDController | None = None
    _throttle_scale: float = 1.0
    _pitch_scale: float = 1.0

    def reset(self) -> None:
        self._throttle_pid = PIDController(
            self.gains.energy_throttle, output_limits=(0.0, 1.0)
        )
        self._pitch_pid = PIDController(
            self.gains.balance_pitch, output_limits=(-1.0, 1.0)
        )
        self._throttle_scale = 1.0
        self._pitch_scale = 1.0

    def adapt_scales(self, *, envelope_active: bool, speed_error: float) -> None:
        if envelope_active:
            self._throttle_scale = max(0.7, self._throttle_scale - 0.03)
            self._pitch_scale = max(0.7, self._pitch_scale - 0.03)
        elif abs(speed_error) > 3.0:
            self._pitch_scale = min(1.4, self._pitch_scale + 0.02)
        else:
            self._throttle_scale = min(1.2, self._throttle_scale + 0.005)

    def step(
        self,
        *,
        altitude_m: float,
        airspeed_m_s: float,
        target_altitude_m: float,
        target_airspeed_m_s: float,
        dt: float,
    ) -> tuple[float, float]:
        if self._throttle_pid is None or self._pitch_pid is None:
            self.reset()
        assert self._throttle_pid is not None and self._pitch_pid is not None

        g = 9.81
        h_err = target_altitude_m - altitude_m
        v_err = target_airspeed_m_s - airspeed_m_s

        specific_energy_err = h_err + (target_airspeed_m_s**2 - airspeed_m_s**2) / (2.0 * g)
        balance_err = h_err - self.speed_weight * v_err

        throttle = self._throttle_pid.step(specific_energy_err * 0.01, dt) * self._throttle_scale
        pitch = self._pitch_pid.step(balance_err * 0.05, dt) * self._pitch_scale
        return (
            float(max(0.0, min(1.0, throttle))),
            float(max(-1.0, min(1.0, pitch))),
        )
