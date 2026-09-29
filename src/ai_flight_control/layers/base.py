from typing import Protocol, runtime_checkable

from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    FlightState,
    ManeuverSetpoint,
    MissionIntent,
    MissionSpec,
)


@runtime_checkable
class MissionController(Protocol):
    def reset(self, spec: MissionSpec) -> None: ...

    def step(self, state: FlightState) -> MissionIntent: ...

    def adapt(self, context: AdaptationContext) -> None: ...


@runtime_checkable
class ManeuverController(Protocol):
    def reset(self) -> None: ...

    def step(self, state: FlightState, intent: MissionIntent) -> ManeuverSetpoint: ...

    def adapt(self, context: AdaptationContext) -> None: ...


@runtime_checkable
class StabilityController(Protocol):
    def reset(self) -> None: ...

    def step(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand: ...

    def adapt(self, context: AdaptationContext) -> None: ...

    @property
    def envelope_active_last_step(self) -> bool: ...


@runtime_checkable
class AdaptivePolicy(Protocol):
    """Online adaptation hook shared by all controllers."""

    def adapt(self, context: AdaptationContext) -> None: ...
