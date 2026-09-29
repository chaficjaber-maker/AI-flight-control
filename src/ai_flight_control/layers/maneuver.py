from dataclasses import dataclass, field

from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    ManeuverSetpoint,
    MissionIntent,
    MissionSegmentKind,
)


@dataclass
class ManeuverLayer:
    """
    Maneuver function (middle layer).

    AI: adapts commanded vertical rate (aggressiveness) from recent errors.
    """

    base_vertical_rate_m_s: float = 3.0
    _rate_gain: float = 1.0

    def reset(self) -> None:
        self._rate_gain = 1.0

    def adapt(self, context: AdaptationContext) -> None:
        if context.envelope_active:
            self._rate_gain = max(0.5, self._rate_gain - 0.05)
        elif abs(context.altitude_error_m) > 30.0:
            self._rate_gain = min(1.4, self._rate_gain + 0.03)
        else:
            self._rate_gain = max(0.7, min(1.2, self._rate_gain * 0.995 + 0.005))

    def step(self, state: FlightState, intent: MissionIntent) -> ManeuverSetpoint:
        seg = intent.segment
        target_alt = seg.target_altitude_m
        alt_err = target_alt - state.altitude_m
        max_rate = seg.max_vertical_rate_m_s * self._rate_gain

        if seg.kind == MissionSegmentKind.HOLD_ALTITUDE:
            desired_rate = max(-max_rate, min(max_rate, 0.35 * alt_err))
            label = "hold"
        elif seg.kind == MissionSegmentKind.CLIMB_TO:
            desired_rate = max(0.0, min(max_rate, 0.5 * alt_err))
            label = "climb"
        else:
            desired_rate = min(0.0, max(-max_rate, 0.5 * alt_err))
            label = "descend"

        return ManeuverSetpoint(
            target_altitude_m=target_alt,
            target_vertical_rate_m_s=desired_rate,
            maneuver_label=label,
        )
