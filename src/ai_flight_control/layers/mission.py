from dataclasses import dataclass, field

from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    MissionIntent,
    MissionSegmentKind,
    MissionSpec,
)


@dataclass
class MissionLayer:
    """
    Mission-following function (outer layer).

    AI: adapts segment completion tolerance from tracking performance.
    """

    altitude_capture_tolerance_m: float = 15.0
    _spec: MissionSpec | None = None
    _segment_index: int = 0
    _adapt_scale: float = 1.0

    def reset(self, spec: MissionSpec) -> None:
        self._spec = spec
        self._segment_index = 0
        self._adapt_scale = 1.0

    def adapt(self, context: AdaptationContext) -> None:
        if abs(context.altitude_error_m) > 40.0:
            self._adapt_scale = min(1.5, self._adapt_scale + 0.02)
        elif abs(context.altitude_error_m) < 5.0:
            self._adapt_scale = max(0.8, self._adapt_scale - 0.01)

    def step(self, state: FlightState) -> MissionIntent:
        if self._spec is None or not self._spec.segments:
            raise RuntimeError("MissionLayer.reset(spec) must be called before step")

        idx = min(self._segment_index, len(self._spec.segments) - 1)
        segment = self._spec.segments[idx]
        tol = self.altitude_capture_tolerance_m * self._adapt_scale
        complete = False

        if segment.kind == MissionSegmentKind.HOLD_ALTITUDE:
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
        )
