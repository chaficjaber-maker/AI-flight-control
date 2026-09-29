from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MissionSegmentKind(str, Enum):
    HOLD_ALTITUDE = "hold_altitude"
    CLIMB_TO = "climb_to"
    DESCEND_TO = "descend_to"
    FLY_TO_WAYPOINT = "fly_to_waypoint"


@dataclass(frozen=True)
class WaypointTarget:
    north_m: float
    east_m: float
    altitude_m: float
    airspeed_m_s: float = 22.0
    capture_radius_m: float = 40.0


@dataclass(frozen=True)
class MissionSegment:
    kind: MissionSegmentKind
    target_altitude_m: float = 0.0
    max_vertical_rate_m_s: float = 5.0
    waypoint: WaypointTarget | None = None

    @staticmethod
    def fly_to(wp: WaypointTarget) -> "MissionSegment":
        return MissionSegment(
            kind=MissionSegmentKind.FLY_TO_WAYPOINT,
            target_altitude_m=wp.altitude_m,
            waypoint=wp,
        )


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
        del initial_m
        return MissionSpec(
            segments=[
                MissionSegment(MissionSegmentKind.CLIMB_TO, target_m),
                MissionSegment(MissionSegmentKind.HOLD_ALTITUDE, target_m),
            ]
        )

    @staticmethod
    def waypoint_route(waypoints: list[WaypointTarget]) -> "MissionSpec":
        return MissionSpec(segments=[MissionSegment.fly_to(wp) for wp in waypoints])


@dataclass(frozen=True)
class FlightState:
    """Fixed-wing state used by all three control layers."""

    altitude_m: float
    airspeed_m_s: float
    north_m: float = 0.0
    east_m: float = 0.0
    heading_rad: float = 0.0
    bank_rad: float = 0.0
    flight_path_angle_rad: float = 0.0
    wind_n_m_s: float = 0.0
    wind_e_m_s: float = 0.0
    time_s: float = 0.0

    @property
    def vertical_velocity_m_s(self) -> float:
        return self.airspeed_m_s * np.sin(self.flight_path_angle_rad)


@dataclass(frozen=True)
class MissionIntent:
    segment_index: int
    segment: MissionSegment
    segment_complete: bool
    bearing_to_target_rad: float = 0.0
    distance_to_target_m: float = 0.0


@dataclass(frozen=True)
class ManeuverSetpoint:
    target_altitude_m: float
    target_airspeed_m_s: float
    target_heading_rad: float
    target_bank_rad: float
    maneuver_label: str
    target_vertical_rate_m_s: float = 0.0


@dataclass(frozen=True)
class ActuatorCommand:
    throttle: float
    pitch: float
    roll: float

    @classmethod
    def from_legacy_thrust(cls, normalized_thrust: float) -> "ActuatorCommand":
        t = float(max(-1.0, min(1.0, normalized_thrust)))
        return cls(throttle=(t + 1.0) * 0.5, pitch=t, roll=0.0)

    @property
    def normalized_thrust(self) -> float:
        return 2.0 * self.throttle - 1.0


@dataclass(frozen=True)
class EnvelopeLimits:
    min_altitude_m: float = 0.0
    max_altitude_m: float = 20_000.0
    min_airspeed_m_s: float = 12.0
    max_airspeed_m_s: float = 35.0
    max_vertical_rate_m_s: float = 8.0
    max_bank_rad: float = np.deg2rad(35.0)
    max_vertical_accel_m_s2: float = 4.0


@dataclass
class AdaptationContext:
    """Shared feedback for online adaptation across layers."""

    altitude_error_m: float = 0.0
    airspeed_error_m_s: float = 0.0
    heading_error_rad: float = 0.0
    vertical_rate_error_m_s: float = 0.0
    cross_track_error_m: float = 0.0
    envelope_active: bool = False
    segment_index: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def wrap_angle_rad(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def bearing_ned_rad(d_north: float, d_east: float) -> float:
    return float(np.arctan2(d_east, d_north))
