from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MissionSegmentKind(str, Enum):
    HOLD_ALTITUDE = "hold_altitude"
    CLIMB_TO = "climb_to"
    DESCEND_TO = "descend_to"


@dataclass(frozen=True)
class MissionSegment:
    kind: MissionSegmentKind
    target_altitude_m: float
    max_vertical_rate_m_s: float = 5.0


@dataclass
class MissionSpec:
    """In-repo mission definition (no GCS). Ordered segments."""

    segments: list[MissionSegment] = field(default_factory=list)

    @staticmethod
    def hold_at(altitude_m: float) -> "MissionSpec":
        return MissionSpec(
            segments=[MissionSegment(MissionSegmentKind.HOLD_ALTITUDE, altitude_m)]
        )

    @staticmethod
    def climb_then_hold(initial_m: float, target_m: float) -> "MissionSpec":
        return MissionSpec(
            segments=[
                MissionSegment(MissionSegmentKind.CLIMB_TO, target_m),
                MissionSegment(MissionSegmentKind.HOLD_ALTITUDE, target_m),
            ]
        )


@dataclass(frozen=True)
class FlightState:
    """Minimal state for longitudinal harness; extend for full fixed-wing."""

    altitude_m: float
    vertical_velocity_m_s: float
    time_s: float = 0.0


@dataclass(frozen=True)
class MissionIntent:
    segment_index: int
    segment: MissionSegment
    segment_complete: bool


@dataclass(frozen=True)
class ManeuverSetpoint:
    target_altitude_m: float
    target_vertical_rate_m_s: float
    maneuver_label: str


@dataclass(frozen=True)
class ActuatorCommand:
    normalized_thrust: float


@dataclass(frozen=True)
class EnvelopeLimits:
    min_altitude_m: float = 0.0
    max_altitude_m: float = 20_000.0
    max_vertical_rate_m_s: float = 8.0
    max_vertical_accel_m_s2: float = 4.0


@dataclass
class AdaptationContext:
    """Shared feedback for online adaptation across layers."""

    altitude_error_m: float = 0.0
    vertical_rate_error_m_s: float = 0.0
    envelope_active: bool = False
    segment_index: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
