from dataclasses import dataclass, field
from typing import Literal

from ai_flight_control.layers.stability import StabilityLayer
from ai_flight_control.policies.stability_policy import NeuralStabilityPolicy
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    EnvelopeLimits,
    FlightState,
    ManeuverSetpoint,
)

PolicyMode = Literal["reference", "neural"]


@dataclass
class StabilityHolder:
    """Stability + critical parameters: reference TECS/PID + neural + guardian."""

    mode: PolicyMode = "reference"
    legacy_vertical_only: bool = False
    limits: EnvelopeLimits = field(default_factory=EnvelopeLimits)
    reference: StabilityLayer = field(default_factory=StabilityLayer)
    neural: NeuralStabilityPolicy = field(default_factory=NeuralStabilityPolicy)

    def reset(self) -> None:
        self.reference.legacy_vertical_only = self.legacy_vertical_only
        self.reference.limits = self.limits
        self.reference.reset()
        self.neural.limits = self.limits
        self.neural.reset()

    def step(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
        if self.legacy_vertical_only or self.mode == "reference":
            return self.reference.step(state, setpoint, dt)
        return self.neural.step(state, setpoint, dt)

    def adapt(self, context: AdaptationContext) -> None:
        if self.legacy_vertical_only or self.mode == "reference":
            self.reference.adapt(context)
        else:
            self.neural.adapt(context)

    @property
    def envelope_active_last_step(self) -> bool:
        if self.legacy_vertical_only or self.mode == "reference":
            return self.reference.envelope_active_last_step
        return self.neural.envelope_active_last_step

    def load_neural_weights(self) -> None:
        self.neural.load_weights()
