import math

import numpy as np

from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec, WaypointTarget


def test_fixed_wing_reaches_first_waypoint():
    spec = MissionSpec.waypoint_route(
        [
            WaypointTarget(
                north_m=800.0,
                east_m=0.0,
                altitude_m=400.0,
                airspeed_m_s=22.0,
                capture_radius_m=100.0,
            ),
            WaypointTarget(north_m=800.0, east_m=600.0, altitude_m=400.0, airspeed_m_s=22.0),
        ]
    )
    result = FlightControlStack().run_fixed_wing(
        spec,
        initial=FixedWingState(
            north_m=0.0,
            east_m=0.0,
            altitude_m=400.0,
            airspeed_m_s=22.0,
            heading_rad=0.0,
            bank_rad=0.0,
            flight_path_angle_rad=0.0,
        ),
        duration_s=90.0,
        dt=0.05,
    )
    min_dist = min(
        math.hypot(log.north_m - 800.0, log.east_m) for log in result.logs
    )
    assert min_dist < 125.0
    assert abs(result.final_altitude_m - 400.0) < 100.0


def test_tecs_climb_segment():
    spec = MissionSpec.climb_then_hold(initial_m=300.0, target_m=500.0)
    result = FlightControlStack().run_fixed_wing(
        spec,
        initial=FixedWingState(
            north_m=0.0,
            east_m=0.0,
            altitude_m=300.0,
            airspeed_m_s=22.0,
            heading_rad=0.0,
            bank_rad=0.0,
            flight_path_angle_rad=0.0,
        ),
        duration_s=180.0,
        dt=0.05,
    )
    assert result.final_altitude_m > 420.0


def test_adaptive_tecs_produces_bounded_commands():
    from ai_flight_control.tecs import AdaptiveTecs

    tecs = AdaptiveTecs()
    tecs.reset()
    throttle, pitch = tecs.step(
        altitude_m=300.0,
        airspeed_m_s=20.0,
        target_altitude_m=500.0,
        target_airspeed_m_s=22.0,
        dt=0.05,
    )
    assert 0.0 <= throttle <= 1.0
    assert -1.0 <= pitch <= 1.0
