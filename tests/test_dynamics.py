from ai_flight_control.dynamics import LongitudinalState, step_longitudinal


def test_full_descent_command_reduces_altitude():
    state = LongitudinalState(altitude_m=100.0, vertical_velocity_m_s=0.0)
    next_state = step_longitudinal(state, normalized_thrust=-1.0, dt=0.1)
    assert next_state.altitude_m < state.altitude_m
    assert next_state.vertical_velocity_m_s < 0.0


def test_positive_thrust_climbs():
    state = LongitudinalState(altitude_m=100.0, vertical_velocity_m_s=0.0)
    next_state = step_longitudinal(state, normalized_thrust=1.0, dt=0.1)
    assert next_state.altitude_m > state.altitude_m
    assert next_state.vertical_velocity_m_s > 0.0
