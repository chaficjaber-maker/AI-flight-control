from ai_flight_control.missions import MissionLibrary
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.simulate import MissionSimulationConfig, run_mission_simulation


def test_run_mission_simulation_reference():
    config = MissionSimulationConfig(
        mission=MissionLibrary.climb_to_cruise(300, 450),
        policy=PolicyConfig.default_reference(),
        duration_s=60.0,
        dt=0.05,
        train_if_missing=False,
    )
    result, report = run_mission_simulation(config)
    assert len(result.logs) > 100
    assert report.final_altitude_m > 300.0
    assert report.mission_segments >= 2
