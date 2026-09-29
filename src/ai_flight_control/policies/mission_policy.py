from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_flight_control.policies.mlp import MLP
from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    MissionIntent,
    MissionSegmentKind,
    MissionSpec,
    bearing_ned_rad,
)

MISSION_INPUT_DIM = 12
MISSION_OUTPUT_DIM = 2
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "mission.npz"


def mission_features(state: FlightState, intent: MissionIntent) -> np.ndarray:
    seg = intent.segment
    wp = seg.waypoint
    alt_tgt = wp.altitude_m if wp else seg.target_altitude_m
    return np.array(
        [
            intent.segment_index / 10.0,
            intent.distance_to_target_m / 1000.0,
            (alt_tgt - state.altitude_m) / 500.0,
            state.airspeed_m_s / 30.0,
            state.vertical_velocity_m_s / 10.0,
            float(seg.kind == MissionSegmentKind.FLY_TO_WAYPOINT),
            float(intent.segment_complete),
            np.sin(intent.bearing_to_target_rad),
            np.cos(intent.bearing_to_target_rad),
            state.altitude_m / 5000.0,
            state.wind_n_m_s / 15.0,
            state.wind_e_m_s / 15.0,
        ],
        dtype=float,
    )


@dataclass
class NeuralMissionPolicy:
    """
    Modulates capture/tolerance scales; segment geometry stays explicit for safety.
    """

    base_altitude_tolerance_m: float = 15.0
    network: MLP = field(
        default_factory=lambda: MLP(MISSION_INPUT_DIM, 16, MISSION_OUTPUT_DIM, seed=3)
    )
    _spec: MissionSpec | None = None
    _segment_index: int = 0
    _tolerance_scale: float = 1.0
    _capture_scale: float = 1.0

    def reset(self, spec: MissionSpec) -> None:
        self._spec = spec
        self._segment_index = 0
        self._tolerance_scale = 1.0
        self._capture_scale = 1.0

    def load_weights(self, path: Path | None = None) -> None:
        self.network = MLP.load(path or DEFAULT_WEIGHTS)

    def _update_scales(self, state: FlightState, intent: MissionIntent) -> None:
        out = self.network.predict(mission_features(state, intent))
        self._tolerance_scale = float(np.clip(1.0 + 0.35 * np.tanh(out[0]), 0.75, 1.5))
        self._capture_scale = float(np.clip(1.0 + 0.35 * np.tanh(out[1]), 0.75, 1.5))

    def adapt(self, context: AdaptationContext) -> None:
        if context.envelope_active:
            self._capture_scale = max(0.75, self._capture_scale - 0.02)
        elif context.cross_track_error_m > 60.0:
            self._capture_scale = min(1.5, self._capture_scale + 0.02)

    def step(self, state: FlightState) -> MissionIntent:
        if self._spec is None or not self._spec.segments:
            raise RuntimeError("NeuralMissionPolicy.reset(spec) must be called before step")

        idx = min(self._segment_index, len(self._spec.segments) - 1)
        segment = self._spec.segments[idx]
        bearing = 0.0
        distance = 0.0

        if segment.kind == MissionSegmentKind.FLY_TO_WAYPOINT and segment.waypoint is not None:
            wp = segment.waypoint
            d_n = wp.north_m - state.north_m
            d_e = wp.east_m - state.east_m
            distance = float(np.hypot(d_n, d_e))
            bearing = bearing_ned_rad(d_n, d_e)

        provisional = MissionIntent(
            segment_index=idx,
            segment=segment,
            segment_complete=False,
            bearing_to_target_rad=bearing,
            distance_to_target_m=distance,
        )
        self._update_scales(state, provisional)

        tol = self.base_altitude_tolerance_m * self._tolerance_scale
        complete = False

        if segment.kind == MissionSegmentKind.FLY_TO_WAYPOINT and segment.waypoint is not None:
            wp = segment.waypoint
            capture = wp.capture_radius_m * self._capture_scale
            alt_ok = abs(state.altitude_m - wp.altitude_m) <= tol
            complete = distance <= capture and alt_ok
        elif segment.kind == MissionSegmentKind.HOLD_ALTITUDE:
            complete = abs(state.altitude_m - segment.target_altitude_m) <= tol
        elif segment.kind in (
            MissionSegmentKind.CLIMB_TO,
            MissionSegmentKind.DESCEND_TO,
        ):
            at_target = abs(state.altitude_m - segment.target_altitude_m) <= tol
            rate_ok = abs(state.vertical_velocity_m_s) < 0.5
            complete = at_target and rate_ok

        if complete and self._segment_index < len(self._spec.segments) - 1:
            self._segment_index += 1
            idx = self._segment_index
            segment = self._spec.segments[idx]
            complete = False

        return MissionIntent(
            segment_index=idx,
            segment=segment,
            segment_complete=complete,
            bearing_to_target_rad=bearing,
            distance_to_target_m=distance,
        )
