from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ai_flight_control.policies.mission_policy import NeuralMissionPolicy
from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    MissionIntent,
    MissionSegmentKind,
    MissionSpec,
    bearing_ned_rad,
)

PolicyMode = Literal["reference", "neural"]


@dataclass
class MissionLayer:
    """
    Mission-following function (outer layer).
    """

    policy_mode: PolicyMode = "reference"
    neural: NeuralMissionPolicy = field(default_factory=NeuralMissionPolicy)
    altitude_capture_tolerance_m: float = 15.0
    _spec: MissionSpec | None = None
    _segment_index: int = 0
    _adapt_scale: float = 1.0

    def reset(self, spec: MissionSpec) -> None:
        self._spec = spec
        self._segment_index = 0
        self._adapt_scale = 1.0
        self.neural.reset(spec)

    def load_neural_weights(self) -> None:
        self.neural.load_weights()

    def adapt(self, context: AdaptationContext) -> None:
        if self.policy_mode == "neural":
            self.neural.adapt(context)
            return
        if abs(context.altitude_error_m) > 40.0 or context.cross_track_error_m > 80.0:
            self._adapt_scale = min(1.5, self._adapt_scale + 0.02)
        elif abs(context.altitude_error_m) < 5.0 and context.cross_track_error_m < 15.0:
            self._adapt_scale = max(0.8, self._adapt_scale - 0.01)

    def step(self, state: FlightState) -> MissionIntent:
        if self.policy_mode == "neural":
            return self.neural.step(state)
        return self._step_reference(state)

    def _step_reference(self, state: FlightState) -> MissionIntent:
        if self._spec is None or not self._spec.segments:
            raise RuntimeError("MissionLayer.reset(spec) must be called before step")

        idx = min(self._segment_index, len(self._spec.segments) - 1)
        segment = self._spec.segments[idx]
        tol = self.altitude_capture_tolerance_m * self._adapt_scale
        complete = False
        bearing = 0.0
        distance = 0.0

        if segment.kind == MissionSegmentKind.FLY_TO_WAYPOINT and segment.waypoint is not None:
            wp = segment.waypoint
            d_n = wp.north_m - state.north_m
            d_e = wp.east_m - state.east_m
            distance = float(np.hypot(d_n, d_e))
            bearing = bearing_ned_rad(d_n, d_e)
            capture = wp.capture_radius_m * self._adapt_scale
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
