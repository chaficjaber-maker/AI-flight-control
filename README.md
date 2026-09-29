# AI-flight-control

In-repo **flight controller development** for fixed-wing UAVs: plant models, control laws, and a small simulation harness to compare controllers—without mission planning, GCS, or autopilot SITL integration (those come later).

## Current scope

**In scope**

- Controller algorithms (PID baseline → learned policies later)
- Minimal dynamics to exercise controllers in tests and CLI runs
- Shared `FlightAgent` / controller interfaces so implementations are swappable
- Regression tests and simple metrics (tracking error, control effort)

**Out of scope (for now)**

- Waypoint missions, flight profiles, and GCS tooling
- ArduPilot / PX4 / JSBSim / MAVLink integration
- Full 6-DOF aerodynamics and hardware-in-the-loop

## What’s included today

- **1-D altitude plant** — simplified vertical channel for fast iteration
- **PID altitude-hold** — classical baseline
- **`PolicyAltitudeAgent`** — same interface; placeholder for ML/RL weights
- **CLI + pytest** — quick A/B runs locally

## Quick start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 -m ai_flight_control.cli --target 1000 --initial 800 --duration 600
```

## Layout

```
src/ai_flight_control/
  dynamics.py      # plant (will grow toward fixed-wing axes)
  pid.py           # reusable PID blocks
  sim.py           # closed-loop harness (not a mission simulator)
  agents/          # controller implementations
tests/
```

## Controller dev roadmap (fixed-wing)

1. **Longitudinal** — airspeed + altitude (TECS-style or decoupled PID baseline)
2. **Lateral** — coordinated turn / roll-to-bank guidance into inner loops
3. **Attitude / rate inner loops** — surface commands with saturations and limits
4. **AI policies** — replace or wrap selected loops via `FlightAgent` and the same harness

Integration with external sims and missions stays a separate phase after controllers are stable here.
