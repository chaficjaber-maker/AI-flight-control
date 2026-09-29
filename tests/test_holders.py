from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.holders.registry import FlightControlHolders
from ai_flight_control.missions import MissionLibrary
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import FlightState, MissionSpec


def test_holders_reference_step_chain():
    holders = FlightControlHolders.default_reference()
    spec = MissionLibrary.climb_to_cruise(300, 450)
    holders.reset(spec)
    state = FlightState(altitude_m=320, airspeed_m_s=22, time_s=0)
    intent = holders.mission.step(state)
    sp = holders.maneuver.step(state, intent)
    cmd = holders.stability.step(state, sp, 0.05)
    assert -1.0 <= cmd.pitch <= 1.0
    assert 0.0 <= cmd.throttle <= 1.0


def test_stack_uses_holders():
    stack = FlightControlStack()
    assert isinstance(stack.holders, FlightControlHolders)
    spec = MissionSpec.hold_at(400)
    result = stack.run_fixed_wing(
        spec,
        initial=FixedWingState(0, 0, 400, 22, 0, 0, 0),
        duration_s=5.0,
        dt=0.05,
    )
    assert result.final_altitude_m > 350


def test_mission_library_box_has_four_legs():
    spec = MissionLibrary.box_pattern()
    assert len(spec.segments) == 4
