from dataclasses import dataclass, field

from ai_flight_control.holders.maneuver_holder import ManeuverHolder
from ai_flight_control.holders.mission_holder import MissionHolder
from ai_flight_control.holders.stability_holder import StabilityHolder
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.state import MissionSpec


@dataclass
class FlightControlHolders:
    """Populated container for the three flight-control functions."""

    mission: MissionHolder = field(default_factory=MissionHolder)
    maneuver: ManeuverHolder = field(default_factory=ManeuverHolder)
    stability: StabilityHolder = field(default_factory=StabilityHolder)

    @classmethod
    def default_reference(cls) -> "FlightControlHolders":
        return cls()

    def apply_policy_config(self, config: PolicyConfig) -> None:
        self.mission.mode = config.mission
        self.maneuver.mode = config.maneuver
        self.stability.mode = config.stability
        if config.stability == "neural":
            self.stability.load_neural_weights()
        if config.maneuver == "neural":
            self.maneuver.load_neural_weights()
        if config.mission == "neural":
            self.mission.load_neural_weights()

    def reset(self, spec: MissionSpec, *, legacy_vertical_only: bool = False) -> None:
        self.stability.legacy_vertical_only = legacy_vertical_only
        self.mission.reset(spec)
        self.maneuver.reset()
        self.stability.reset()
