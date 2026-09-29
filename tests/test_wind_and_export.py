import json
import math

import numpy as np

from ai_flight_control.fixed_wing_plant import FixedWingControls, FixedWingState, step_fixed_wing
from ai_flight_control.policies.export_embedded import export_all_from_npz, forward_embedded_json
from ai_flight_control.policies.mlp import MLP
from ai_flight_control.policies.train import train_all_in_order
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec, WaypointTarget
from ai_flight_control.wind import WindField, ground_velocity_ned


def test_wind_changes_ground_track():
    calm = FixedWingState(0, 0, 400, 22, 0, 0, 0)
    windy = step_fixed_wing(
        calm,
        FixedWingControls(0.7, 0.0, 0.0),
        1.0,
        wind=WindField(east_m_s=8.0),
    )
    no_wind = step_fixed_wing(
        calm,
        FixedWingControls(0.7, 0.0, 0.0),
        1.0,
        wind=WindField.calm(),
    )
    assert windy.east_m > no_wind.east_m + 5.0


def test_ground_velocity_includes_wind():
    gn, ge, _ = ground_velocity_ned(20.0, 0.0, 0.0, 0.0, 5.0)
    assert ge >= 5.0
    assert gn >= 20.0


def test_embedded_export_matches_numpy_forward(tmp_path):
    net = MLP(4, 6, 2, seed=0)
    x = np.array([0.1, -0.2, 0.3, 0.0])
    expected = net.predict(x)
    np.savez(
        tmp_path / "stability.npz",
        w1=net.weights.w1,
        b1=net.weights.b1,
        w2=net.weights.w2,
        b2=net.weights.b2,
        input_dim=4,
        hidden_dim=6,
        output_dim=2,
    )
    from ai_flight_control.policies.export_embedded import export_mlp_c_header, export_mlp_json

    out_dir = tmp_path / "embedded"
    json_path = export_mlp_json(net, "stability", out_dir)
    header = export_mlp_c_header(net, "stability", out_dir)
    got = forward_embedded_json(json_path, x)
    assert np.allclose(got, expected, atol=1e-5)
    assert header.exists()
    assert "stability_forward" in header.read_text()


def test_neural_stack_under_crosswind():
    train_all_in_order(quick=True)
    stack = FlightControlStack.neural_all()
    spec = MissionSpec.waypoint_route(
        [WaypointTarget(500, 0, 400, 22, capture_radius_m=120)]
    )
    result = stack.run_fixed_wing(
        spec,
        initial=FixedWingState(0, 0, 400, 22, 0, 0, 0),
        duration_s=80.0,
        dt=0.05,
        wind=WindField(north_m_s=4.0, east_m_s=-3.0, gust_std_m_s=0.5),
    )
    min_dist = min(math.hypot(log.north_m - 500, log.east_m) for log in result.logs)
    assert min_dist < 200.0
