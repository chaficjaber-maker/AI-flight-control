"""Predefined mission specs for sims, training, and regression."""

from ai_flight_control.state import MissionSegment, MissionSegmentKind, MissionSpec, WaypointTarget


class MissionLibrary:
    @staticmethod
    def hold_cruise(altitude_m: float = 400.0) -> MissionSpec:
        return MissionSpec.hold_at(altitude_m)

    @staticmethod
    def climb_to_cruise(from_m: float, to_m: float) -> MissionSpec:
        return MissionSpec.climb_then_hold(from_m, to_m)

    @staticmethod
    def box_pattern(
        *,
        center_n: float = 500.0,
        center_e: float = 0.0,
        leg_m: float = 400.0,
        altitude_m: float = 450.0,
        airspeed_m_s: float = 22.0,
    ) -> MissionSpec:
        return MissionSpec.waypoint_route(
            [
                WaypointTarget(center_n + leg_m, center_e, altitude_m, airspeed_m_s, 90),
                WaypointTarget(center_n + leg_m, center_e + leg_m, altitude_m, airspeed_m_s, 90),
                WaypointTarget(center_n, center_e + leg_m, altitude_m, airspeed_m_s, 90),
                WaypointTarget(center_n, center_e, altitude_m, airspeed_m_s, 90),
            ]
        )

    @staticmethod
    def training_route() -> MissionSpec:
        return MissionSpec(
            segments=[
                MissionSegment(MissionSegmentKind.CLIMB_TO, 500.0),
                MissionSegment.fly_to(WaypointTarget(700, 0, 500, 22, 100)),
                MissionSegment.fly_to(WaypointTarget(900, 250, 520, 23, 100)),
                MissionSegment(MissionSegmentKind.HOLD_ALTITUDE, 520.0),
            ]
        )
