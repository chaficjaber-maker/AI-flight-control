import math

from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.policies.mlp import MLP, train_supervised
from ai_flight_control.policies.dataset import DatasetConfig, collect_stability_dataset
from ai_flight_control.policies.train import ensure_trained_weights, train_all_in_order
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec, WaypointTarget


def test_train_stability_mlp_reduces_loss():
    x, y = collect_stability_dataset(DatasetConfig(stability_samples=120, seed=99))
    net = MLP(12, 16, 3, seed=99)
    losses = train_supervised(net, x, y, epochs=30, learning_rate=0.02, batch_size=32)
    assert losses[-1] < losses[0]


def test_neural_stability_closed_loop():
    ensure_trained_weights(quick=True)
    stack = FlightControlStack.from_policy_config(
        PolicyConfig(stability="neural", maneuver="reference", mission="reference"),
        train_if_missing=False,
    )
    spec = MissionSpec.climb_then_hold(300.0, 500.0)
    result = stack.run_fixed_wing(
        spec,
        initial=FixedWingState(0, 0, 300, 22, 0, 0, 0),
        duration_s=150.0,
        dt=0.05,
    )
    assert result.final_altitude_m > 350.0


def test_full_neural_stack_after_ordered_training():
    train_all_in_order(quick=True)
    stack = FlightControlStack.from_policy_config(
        PolicyConfig.all_neural(),
        train_if_missing=False,
    )
    spec = MissionSpec.waypoint_route(
        [WaypointTarget(500, 0, 400, 22, capture_radius_m=100)]
    )
    result = stack.run_fixed_wing(
        spec,
        initial=FixedWingState(0, 0, 400, 22, 0, 0, 0),
        duration_s=70.0,
        dt=0.05,
    )
    min_dist = min(math.hypot(log.north_m - 500, log.east_m) for log in result.logs)
    assert min_dist < 320.0
