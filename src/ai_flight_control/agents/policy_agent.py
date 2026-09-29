from dataclasses import dataclass, field

from ai_flight_control.agents.pid_agent import PIDAltitudeAgent
from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.pid import PIDGains


@dataclass
class PolicyAltitudeAgent:
    """
    Placeholder for ML/RL policies.

    Until a trained model is wired in, delegates to a tuned PID baseline so
    simulations and tests stay runnable.
    """

    fallback_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.015, ki=0.002, kd=0.08)
    )
    _fallback: PIDAltitudeAgent | None = None

    def reset(self) -> None:
        if self._fallback is None:
            self._fallback = PIDAltitudeAgent(gains=self.fallback_gains)
        self._fallback.reset()

    def act(
        self,
        state: LongitudinalState,
        target_altitude_m: float,
        dt: float,
    ) -> float:
        if self._fallback is None:
            self.reset()
        assert self._fallback is not None
        return self._fallback.act(state, target_altitude_m, dt)
