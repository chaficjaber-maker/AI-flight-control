"""Behavioral-cloning trainers: reference teachers -> NumPy MLP weights."""

from __future__ import annotations

from pathlib import Path

from ai_flight_control.policies.dataset import (
    DatasetConfig,
    collect_maneuver_dataset,
    collect_mission_dataset,
    collect_stability_dataset,
)
from ai_flight_control.policies.maneuver_policy import (
    MANEUVER_INPUT_DIM,
    MANEUVER_OUTPUT_DIM,
)
from ai_flight_control.policies.mission_policy import MISSION_INPUT_DIM, MISSION_OUTPUT_DIM
from ai_flight_control.policies.mlp import MLP, train_supervised
from ai_flight_control.policies.stability_policy import (
    STABILITY_INPUT_DIM,
    STABILITY_OUTPUT_DIM,
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def ensure_trained_weights(*, quick: bool = True) -> dict[str, Path]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    names = ("stability.npz", "maneuver.npz", "mission.npz")
    expected = {
        "stability.npz": STABILITY_INPUT_DIM,
        "maneuver.npz": MANEUVER_INPUT_DIM,
        "mission.npz": MISSION_INPUT_DIM,
    }
    if all((WEIGHTS_DIR / n).exists() for n in names):
        try:
            for n, dim in expected.items():
                net = MLP.load(WEIGHTS_DIR / n)
                if net.input_dim != dim:
                    raise ValueError("stale weights")
            return {n.replace(".npz", ""): WEIGHTS_DIR / n for n in names}
        except Exception:
            pass
    return train_all_in_order(quick=quick)


def train_stability_policy(
    *,
    epochs: int = 120,
    seed: int = 1,
    config: DatasetConfig | None = None,
) -> Path:
    cfg = config or DatasetConfig(stability_samples=2500 if epochs > 50 else 400, seed=seed)
    x, y = collect_stability_dataset(cfg)
    net = MLP(STABILITY_INPUT_DIM, 32, STABILITY_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.006, batch_size=128)
    out = WEIGHTS_DIR / "stability.npz"
    net.save(out)
    return out


def train_maneuver_policy(
    *,
    epochs: int = 120,
    seed: int = 2,
    config: DatasetConfig | None = None,
) -> Path:
    cfg = config or DatasetConfig(maneuver_samples=2500 if epochs > 50 else 400, seed=seed + 1)
    x, y = collect_maneuver_dataset(cfg)
    net = MLP(MANEUVER_INPUT_DIM, 40, MANEUVER_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.006, batch_size=128)
    out = WEIGHTS_DIR / "maneuver.npz"
    net.save(out)
    return out


def train_mission_policy(
    *,
    epochs: int = 80,
    seed: int = 3,
    config: DatasetConfig | None = None,
) -> Path:
    cfg = config or DatasetConfig(mission_samples=1800 if epochs > 50 else 300, seed=seed + 2)
    x, y = collect_mission_dataset(cfg)
    net = MLP(MISSION_INPUT_DIM, 24, MISSION_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.008, batch_size=128)
    out = WEIGHTS_DIR / "mission.npz"
    net.save(out)
    return out


def train_all_in_order(*, quick: bool = False) -> dict[str, Path]:
    if quick:
        cfg = DatasetConfig(
            stability_samples=350,
            maneuver_samples=350,
            mission_samples=250,
        )
        epochs_s, epochs_m, epochs_mi = 35, 35, 25
    else:
        cfg = DatasetConfig()
        epochs_s, epochs_m, epochs_mi = 140, 140, 90
    paths = {
        "stability": train_stability_policy(epochs=epochs_s, config=cfg),
        "maneuver": train_maneuver_policy(epochs=epochs_m, config=cfg),
        "mission": train_mission_policy(epochs=epochs_mi, config=cfg),
    }
    from ai_flight_control.policies.export_embedded import export_all_from_npz

    export_all_from_npz()
    return paths


def smoke_test_neural_stack():
    from ai_flight_control.fixed_wing_plant import FixedWingState
    from ai_flight_control.stack import FlightControlStack
    from ai_flight_control.state import MissionSpec, WaypointTarget

    train_all_in_order(quick=True)
    stack = FlightControlStack.neural_all()
    spec = MissionSpec.waypoint_route(
        [WaypointTarget(600, 0, 400, 22, capture_radius_m=100)]
    )
    return stack.run_fixed_wing(
        spec,
        initial=FixedWingState(0, 0, 400, 22, 0, 0, 0),
        duration_s=60.0,
        dt=0.05,
    )
