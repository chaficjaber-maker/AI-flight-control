from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDController:
    def __init__(self, gains: PIDGains, output_limits: tuple[float, float] = (-1.0, 1.0)):
        self.gains = gains
        self.output_limits = output_limits
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")

        self._integral += error * dt
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        output = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        lo, hi = self.output_limits
        return max(lo, min(hi, output))
