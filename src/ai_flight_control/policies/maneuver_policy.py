from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_flight_control.policies.mlp import MLP
from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    ManeuverSetpoint,
    MissionIntent,
    MissionSegmentKind,
    bearing_ned_rad,
    wrap_angle_rad,
)

MANEUVER_INPUT_DIM = 12
MANEUVER_OUTPUT_DIM = 6
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "maneuver.npz"


def maneuver_features(state: FlightState, intent: MissionIntent) -> np.ndarray:
    seg = intent.segment
    wp = seg.waypoint
    d_n = (wp.north_m - state.north_m) / 1000.0 if wp else 0.0
    d_e = (wp.east_m - state.east_m) / 1000.0 if wp else 0.0
    kind = seg.kind
    return np.array(
        [
            state.altitude_m / 5000.0,
            state.airspeed_m_s / 30.0,
            np.sin(state.heading_rad),
            np.cos(state.heading_rad),
            d_n,
            d_e,
            intent.distance_to_target_m / 1000.0,
            float(kind == MissionSegmentKind.FLY_TO_WAYPOINT),
            float(kind == MissionSegmentKind.CLIMB_TO),
            float(kind == MissionSegmentKind.HOLD_ALTITUDE),
            seg.target_altitude_m / 5000.0,
            intent.segment_index / 10.0,
        ],
        dtype=float,
    )


def decode_maneuver_output(
    out: np.ndarray,
    state: FlightState,
    intent: MissionIntent,
) -> ManeuverSetpoint:
    seg = intent.segment
    wp = seg.waypoint
    target_alt = seg.target_altitude_m + float(out[0]) * 100.0
    target_speed = state.airspeed_m_s + float(out[1]) * 5.0
    if wp is not None:
        target_alt = wp.altitude_m + float(out[0]) * 50.0
        target_speed = wp.airspeed_m_s + float(out[1]) * 3.0

    heading = state.heading_rad + float(out[2]) * np.pi
    if wp is not None:
        heading = bearing_ned_rad(wp.north_m - state.north_m, wp.east_m - state.east_m)
        heading = heading + float(out[2]) * 0.15

    bank = float(np.tanh(out[3])) * np.deg2rad(25.0)
    v_rate = float(np.tanh(out[4])) * 5.0
    label_idx = int(np.argmax(out[5:6])) if len(out) > 5 else 0
    labels = ("hold", "climb", "track", "level_turn")
    label = labels[min(label_idx, len(labels) - 1)]
    if wp is not None:
        hdg_err = abs(wrap_angle_rad(heading - state.heading_rad))
        label = "level_turn" if hdg_err > np.deg2rad(8.0) else "track"

    return ManeuverSetpoint(
        target_altitude_m=target_alt,
        target_airspeed_m_s=max(12.0, min(35.0, target_speed)),
        target_heading_rad=heading,
        target_bank_rad=bank,
        maneuver_label=label,
        target_vertical_rate_m_s=v_rate,
    )


@dataclass
class NeuralManeuverPolicy:
    network: MLP = field(
        default_factory=lambda: MLP(MANEUVER_INPUT_DIM, 32, MANEUVER_OUTPUT_DIM, seed=2)
    )
    _online_lr: float = 0.0004

    def reset(self) -> None:
        pass

    def load_weights(self, path: Path | None = None) -> None:
        self.network = MLP.load(path or DEFAULT_WEIGHTS)

    def step(self, state: FlightState, intent: MissionIntent) -> ManeuverSetpoint:
        features = maneuver_features(state, intent)
        out = self.network.predict(features)
        return decode_maneuver_output(out, state, intent)

    def adapt(self, context: AdaptationContext) -> None:
        del context
