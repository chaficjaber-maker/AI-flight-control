from dataclasses import dataclass, field
from typing import Literal

from ai_flight_control.layers.mission import MissionLayer
from ai_flight_control.policies.mission_policy import NeuralMissionPolicy
from ai_flight_control.state import AdaptationContext, FlightState, MissionIntent, MissionSpec

PolicyMode = Literal["reference", "neural"]


@dataclass
class MissionHolder:
    """Mission-following function: reference rules + neural policy slot."""

    mode: PolicyMode = "reference"
    reference: MissionLayer = field(default_factory=MissionLayer)
    neural: NeuralMissionPolicy = field(default_factory=NeuralMissionPolicy)

    def reset(self, spec: MissionSpec) -> None:
        self.reference.reset(spec)
        self.neural.reset(spec)

    def step(self, state: FlightState) -> MissionIntent:
        if self.mode == "neural":
            return self.neural.step(state)
        return self.reference.step(state)

    def adapt(self, context: AdaptationContext) -> None:
        if self.mode == "neural":
            self.neural.adapt(context)
        else:
            self.reference.adapt(context)

    def load_neural_weights(self) -> None:
        self.neural.load_weights()
