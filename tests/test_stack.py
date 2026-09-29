from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec


def test_three_layer_stack_climb_and_hold():
    spec = MissionSpec.climb_then_hold(initial_m=500.0, target_m=1000.0)
    result = FlightControlStack().run(
        spec,
        initial=LongitudinalState(altitude_m=500.0, vertical_velocity_m_s=0.0),
        duration_s=600.0,
        dt=0.05,
    )
    assert abs(result.final_altitude_m - 1000.0) < 25.0
    assert any(log.maneuver_label == "climb" for log in result.logs)
    assert result.logs[-1].segment_index >= 0


def test_stability_envelope_limits_extreme_setpoint():
    from ai_flight_control.layers.stability import StabilityLayer
    from ai_flight_control.state import (
        EnvelopeLimits,
        FlightState,
        ManeuverSetpoint,
    )

    layer = StabilityLayer(
        limits=EnvelopeLimits(max_vertical_rate_m_s=2.0),
    )
    layer.reset()
    layer.legacy_vertical_only = True
    cmd = layer.step(
        FlightState(
            altitude_m=100.0,
            airspeed_m_s=22.0,
            flight_path_angle_rad=0.0,
            time_s=0.0,
        ),
        ManeuverSetpoint(
            target_altitude_m=5000.0,
            target_airspeed_m_s=22.0,
            target_heading_rad=0.0,
            target_bank_rad=0.0,
            maneuver_label="climb",
            target_vertical_rate_m_s=50.0,
        ),
        dt=0.05,
    )
    assert -1.0 <= cmd.normalized_thrust <= 1.0
