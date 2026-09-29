"""Behavioral-cloning trainers: reference teachers -> NumPy MLP weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_flight_control.layers.maneuver import ManeuverLayer
from ai_flight_control.layers.mission import MissionLayer
from ai_flight_control.layers.stability import StabilityLayer
from ai_flight_control.policies.maneuver_policy import (
    MANEUVER_INPUT_DIM,
    MANEUVER_OUTPUT_DIM,
    maneuver_features,
)
from ai_flight_control.policies.mission_policy import (
    MISSION_INPUT_DIM,
    MISSION_OUTPUT_DIM,
    mission_features,
)
from ai_flight_control.policies.mlp import MLP, train_supervised
from ai_flight_control.policies.stability_policy import (
    STABILITY_INPUT_DIM,
    STABILITY_OUTPUT_DIM,
    stability_features,
)
from ai_flight_control.state import FlightState, MissionSpec, WaypointTarget

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def ensure_trained_weights(*, quick: bool = True) -> dict[str, Path]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    names = ("stability.npz", "maneuver.npz", "mission.npz")
    if all((WEIGHTS_DIR / n).exists() for n in names):
        return {n.replace(".npz", ""): WEIGHTS_DIR / n for n in names}
    return train_all_in_order(quick=quick)


def _collect_stability_dataset(samples: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    teacher = StabilityLayer()
    teacher.legacy_vertical_only = False
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _ in range(samples):
        state = FlightState(
            altitude_m=float(rng.uniform(200, 1200)),
            airspeed_m_s=float(rng.uniform(16, 28)),
            north_m=float(rng.uniform(0, 1500)),
            east_m=float(rng.uniform(-200, 200)),
            heading_rad=float(rng.uniform(-np.pi, np.pi)),
            bank_rad=float(rng.uniform(-0.4, 0.4)),
            flight_path_angle_rad=float(rng.uniform(-0.12, 0.12)),
        )
        from ai_flight_control.state import ManeuverSetpoint

        sp = ManeuverSetpoint(
            target_altitude_m=float(rng.uniform(200, 1200)),
            target_airspeed_m_s=float(rng.uniform(18, 26)),
            target_heading_rad=state.heading_rad + float(rng.uniform(-0.8, 0.8)),
            target_bank_rad=float(rng.uniform(-0.35, 0.35)),
            maneuver_label="track",
        )
        teacher.reset()
        cmd = teacher.step(state, sp, dt=0.05)
        feat = stability_features(state, sp)
        t = float(np.clip(cmd.throttle, 1e-4, 1.0 - 1e-4))
        throttle_logit = float(np.log(t / (1.0 - t)))
        pitch_raw = float(np.arctanh(np.clip(cmd.pitch, -0.999, 0.999)))
        roll_raw = float(np.arctanh(np.clip(cmd.roll, -0.999, 0.999)))
        xs.append(feat)
        ys.append(np.array([throttle_logit, pitch_raw, roll_raw]))
    return np.stack(xs), np.stack(ys)


def train_stability_policy(*, epochs: int = 120, seed: int = 1) -> Path:
    x, y = _collect_stability_dataset(seed=seed)
    net = MLP(STABILITY_INPUT_DIM, 24, STABILITY_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.008)
    out = WEIGHTS_DIR / "stability.npz"
    net.save(out)
    return out


def _collect_maneuver_dataset(samples: int = 800, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    teacher = ManeuverLayer()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _ in range(samples):
        from ai_flight_control.state import MissionIntent, MissionSegment

        wp = WaypointTarget(
            north_m=float(rng.uniform(200, 2000)),
            east_m=float(rng.uniform(-500, 500)),
            altitude_m=float(rng.uniform(300, 800)),
            airspeed_m_s=22.0,
        )
        state = FlightState(
            altitude_m=float(rng.uniform(250, 750)),
            airspeed_m_s=float(rng.uniform(18, 26)),
            north_m=float(rng.uniform(0, 500)),
            east_m=float(rng.uniform(-300, 300)),
            heading_rad=float(rng.uniform(-np.pi, np.pi)),
        )
        seg = MissionSegment.fly_to(wp)
        intent = MissionIntent(
            segment_index=0,
            segment=seg,
            segment_complete=False,
            bearing_to_target_rad=float(rng.uniform(-np.pi, np.pi)),
            distance_to_target_m=float(rng.uniform(50, 1500)),
        )
        teacher.reset()
        sp = teacher.step(state, intent)
        feat = maneuver_features(state, intent)
        xs.append(feat)
        ys.append(
            np.array(
                [
                    (sp.target_altitude_m - (wp.altitude_m if wp else 0)) / 50.0,
                    (sp.target_airspeed_m_s - 22.0) / 3.0,
                    0.0,
                    sp.target_bank_rad / np.deg2rad(25.0),
                    sp.target_vertical_rate_m_s / 5.0,
                    0.0,
                ]
            )
        )
    return np.stack(xs), np.stack(ys)


def train_maneuver_policy(*, epochs: int = 120, seed: int = 2) -> Path:
    x, y = _collect_maneuver_dataset(seed=seed)
    net = MLP(MANEUVER_INPUT_DIM, 32, MANEUVER_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.008)
    out = WEIGHTS_DIR / "maneuver.npz"
    net.save(out)
    return out


def _collect_mission_dataset(samples: int = 600, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    teacher = MissionLayer()
    spec = MissionSpec.waypoint_route(
        [WaypointTarget(500, 0, 400, 22), WaypointTarget(900, 200, 450, 22)]
    )
    teacher.reset(spec)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _ in range(samples):
        airspeed = float(rng.uniform(18, 26))
        gamma = float(rng.uniform(-0.12, 0.12))
        state = FlightState(
            altitude_m=float(rng.uniform(350, 500)),
            airspeed_m_s=airspeed,
            north_m=float(rng.uniform(0, 900)),
            east_m=float(rng.uniform(-100, 250)),
            heading_rad=float(rng.uniform(-np.pi, np.pi)),
            flight_path_angle_rad=gamma,
        )
        intent = teacher.step(state)
        feat = mission_features(state, intent)
        wp = intent.segment.waypoint
        alt_tgt = wp.altitude_m if wp else intent.segment.target_altitude_m
        alt_err = abs(state.altitude_m - alt_tgt)
        tol_out = float(np.tanh((alt_err - 15.0) / 25.0))
        cap_out = float(np.tanh((intent.distance_to_target_m - 80.0) / 120.0))
        xs.append(feat)
        ys.append(np.array([tol_out, cap_out]))
    return np.stack(xs), np.stack(ys)


def train_mission_policy(*, epochs: int = 80, seed: int = 3) -> Path:
    x, y = _collect_mission_dataset(seed=seed)
    net = MLP(MISSION_INPUT_DIM, 16, MISSION_OUTPUT_DIM, seed=seed)
    train_supervised(net, x, y, epochs=epochs, learning_rate=0.01)
    out = WEIGHTS_DIR / "mission.npz"
    net.save(out)
    return out


def train_all_in_order(*, quick: bool = False) -> dict[str, Path]:
    epochs = 40 if quick else 120
    paths = {
        "stability": train_stability_policy(epochs=epochs),
        "maneuver": train_maneuver_policy(epochs=epochs),
        "mission": train_mission_policy(epochs=40 if quick else 80),
    }
    return paths


def smoke_test_neural_stack():
    from ai_flight_control.fixed_wing_plant import FixedWingState
    from ai_flight_control.stack import FlightControlStack

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
