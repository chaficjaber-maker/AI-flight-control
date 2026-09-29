from dataclasses import dataclass, field
from typing import Literal

from ai_flight_control.layers.maneuver import ManeuverLayer
from ai_flight_control.policies.maneuver_policy import NeuralManeuverPolicy
from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    ManeuverSetpoint,
    MissionIntent,
)

PolicyMode = Literal["reference", "neural"]


@dataclass
class ManeuverHolder:
    """Maneuver function: reference profiles + neural policy slot."""

    mode: PolicyMode = "reference"
    reference: ManeuverLayer = field(default_factory=ManeuverLayer)
    neural: NeuralManeuverPolicy = field(default_factory=NeuralManeuverPolicy)
    _adapt_gain: float = 1.0

    def reset(self) -> None:
        self.reference.reset()
        self.neural.reset()
        self._adapt_gain = 1.0

    def step(self, state: FlightState, intent: MissionIntent) -> ManeuverSetpoint:
        if self.mode == "neural":
            sp = self.neural.step(state, intent)
            return self._apply_adapt_gain(sp)
        return self.reference.step(state, intent)

    def _apply_adapt_gain(self, sp: ManeuverSetpoint) -> ManeuverSetpoint:
        if abs(self._adapt_gain - 1.0) < 0.01:
            return sp
        return ManeuverSetpoint(
            target_altitude_m=sp.target_altitude_m,
            target_airspeed_m_s=sp.target_airspeed_m_s,
            target_heading_rad=sp.target_heading_rad,
            target_bank_rad=sp.target_bank_rad * self._adapt_gain,
            maneuver_label=sp.maneuver_label,
            target_vertical_rate_m_s=sp.target_vertical_rate_m_s * self._adapt_gain,
        )

    def adapt(self, context: AdaptationContext) -> None:
        if self.mode == "neural":
            self.neural.adapt(context)
            if context.envelope_active:
                self._adapt_gain = max(0.6, self._adapt_gain - 0.03)
            elif abs(context.heading_error_rad) > 0.35:
                self._adapt_gain = min(1.25, self._adapt_gain + 0.02)
            return
        self.reference.adapt(context)

    def load_neural_weights(self) -> None:
        self.neural.load_weights()
