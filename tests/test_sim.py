from ai_flight_control.agents import PIDAltitudeAgent
from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.sim import run_altitude_hold


def test_pid_reaches_target_within_reasonable_error():
    result = run_altitude_hold(
        PIDAltitudeAgent(),
        initial=LongitudinalState(altitude_m=500.0, vertical_velocity_m_s=0.0),
        target_altitude_m=1000.0,
        duration_s=600.0,
        dt=0.05,
    )
    final_error = abs(result.altitudes_m[-1] - result.target_altitude_m)
    assert final_error < 10.0
