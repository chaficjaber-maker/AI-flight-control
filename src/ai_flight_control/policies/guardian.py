from dataclasses import dataclass

from ai_flight_control.state import ActuatorCommand, EnvelopeLimits, FlightState


@dataclass
class GuardianResult:
    command: ActuatorCommand
    envelope_active: bool


def apply_stability_guardian(
    raw: ActuatorCommand,
    state: FlightState,
    limits: EnvelopeLimits,
) -> GuardianResult:
    """Hard envelope limits applied after any AI stability policy."""
    events = 0
    throttle = raw.throttle
    pitch = raw.pitch
    roll = raw.roll

    if state.airspeed_m_s <= limits.min_airspeed_m_s * 1.05:
        throttle = max(throttle, 0.55)
        pitch = max(pitch, -0.2)
        events += 1

    if state.altitude_m <= limits.min_altitude_m:
        pitch = max(pitch, 0.0)
        events += 1
    if state.altitude_m >= limits.max_altitude_m:
        pitch = min(pitch, 0.0)
        throttle = min(throttle, 0.7)
        events += 1

    if abs(roll) >= 0.98:
        events += 1

    return GuardianResult(
        command=ActuatorCommand(
            throttle=float(max(0.0, min(1.0, throttle))),
            pitch=float(max(-1.0, min(1.0, pitch))),
            roll=float(max(-1.0, min(1.0, roll))),
        ),
        envelope_active=events > 0,
    )
