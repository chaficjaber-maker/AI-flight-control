from dataclasses import dataclass

import numpy as np

from ai_flight_control.state import (
    AdaptationContext,
    FlightState,
    ManeuverSetpoint,
    MissionIntent,
    MissionSegmentKind,
    bearing_ned_rad,
    wrap_angle_rad,
)


@dataclass
class ManeuverLayer:
    """
    Maneuver function (middle layer).

    Longitudinal profiles + coordinated level-turn heading capture.
    """

    base_vertical_rate_m_s: float = 3.0
    max_bank_rad: float = np.deg2rad(25.0)
    _rate_gain: float = 1.0
    _turn_gain: float = 1.0

    def reset(self) -> None:
        self._rate_gain = 1.0
        self._turn_gain = 1.0

    def adapt(self, context: AdaptationContext) -> None:
        if context.envelope_active:
            self._rate_gain = max(0.5, self._rate_gain - 0.05)
            self._turn_gain = max(0.5, self._turn_gain - 0.05)
        elif abs(context.altitude_error_m) > 30.0:
            self._rate_gain = min(1.4, self._rate_gain + 0.03)
        elif abs(context.heading_error_rad) > np.deg2rad(20.0):
            self._turn_gain = min(1.3, self._turn_gain + 0.02)

    def step(self, state: FlightState, intent: MissionIntent) -> ManeuverSetpoint:
        seg = intent.segment
        wp = seg.waypoint

        if seg.kind == MissionSegmentKind.FLY_TO_WAYPOINT and wp is not None:
            return self._waypoint_maneuver(state, intent, wp)

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
            target_airspeed_m_s=state.airspeed_m_s,
            target_heading_rad=state.heading_rad,
            target_bank_rad=0.0,
            maneuver_label=label,
            target_vertical_rate_m_s=desired_rate,
        )

    def _waypoint_maneuver(self, state: FlightState, intent: MissionIntent, wp) -> ManeuverSetpoint:
        d_n = wp.north_m - state.north_m
        d_e = wp.east_m - state.east_m
        bearing = bearing_ned_rad(d_n, d_e)
        hdg_err = wrap_angle_rad(bearing - state.heading_rad)

        bank = max(
            -self.max_bank_rad,
            min(self.max_bank_rad, hdg_err * 1.1 * self._turn_gain),
        )
        label = "level_turn" if abs(hdg_err) > np.deg2rad(8.0) else "track"

        alt_err = wp.altitude_m - state.altitude_m
        return ManeuverSetpoint(
            target_altitude_m=wp.altitude_m,
            target_airspeed_m_s=wp.airspeed_m_s,
            target_heading_rad=bearing,
            target_bank_rad=bank,
            maneuver_label=label,
            target_vertical_rate_m_s=max(-3.0, min(3.0, 0.25 * alt_err)),
        )
