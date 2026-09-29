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

## Architecture (three AI-adaptive functions)

| Layer | Function | Reference module |
|-------|----------|------------------|
| **Mission** | Follow operational plan (segments, sequencing) | `layers/mission.py` |
| **Maneuver** | Execute climb/hold/etc. setpoints | `layers/maneuver.py` |
| **Stability** | Critical parameters, inner loop, envelope guardian | `layers/stability.py` |

Orchestration: `FlightControlStack` in `stack.py`. Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Each layer implements **`adapt()`** for online tuning (reference rules today; swap in neural policies per layer later).

## Layout

```
src/ai_flight_control/
  holders/         # MissionHolder, ManeuverHolder, StabilityHolder (+ registry)
  layers/          # reference controllers (TECS, turns, waypoint logic)
  policies/        # neural MLP policies + training + embedded export
  missions/        # MissionLibrary (hold, climb, box, training route)
  stack.py         # FlightControlStack composes holders
  state.py         # shared types
tests/
```

### CLI

```bash
python3 -m ai_flight_control.cli stack --mission-name training --stability neural
python3 -m ai_flight_control.cli stack --mission-file missions/examples/box.json
python3 -m ai_flight_control.cli legacy --target 1000 --agent pid
```

Mission JSON supports either a `segments` array (full control) or a `waypoints` array (fly-to sequence). See `missions/examples/`.

## Fixed-wing harness

```python
from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec, WaypointTarget

spec = MissionSpec.waypoint_route([
    WaypointTarget(north_m=800, east_m=0, altitude_m=400, airspeed_m_s=22),
])
FlightControlStack().run_fixed_wing(
    spec,
    initial=FixedWingState(0, 0, 400, 22, 0, 0, 0),
    duration_s=90,
    dt=0.05,
)
```

- **Stability:** adaptive TECS (height + speed) + bank inner loop + envelope limits  
- **Maneuver:** level turn / track to waypoint  
- **Mission:** waypoint sequencing with capture radius  

Legacy 1-D: `stack.run(...)` with `LongitudinalState` (unchanged tests).

## Neural policies (stability → maneuver → mission)

```bash
python3 scripts/train_policies.py   # rich BC datasets -> policies/weights/*.npz
python3 scripts/export_policies.py  # JSON + C headers -> policies/weights/embedded/
```

**Wind:** pass `WindField` to `run_fixed_wing(..., wind=WindField(north_m_s=4, east_m_s=-3))`.  
Policy inputs include normalized wind components; BC datasets sample calm, steady, and gusty wind.

```python
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.stack import FlightControlStack

# Rollout order: stability first, then enable maneuver + mission neural modes
stack = FlightControlStack.from_policy_config(
    PolicyConfig(stability="neural", maneuver="neural", mission="neural"),
)
```

Stability outputs pass through a **non-learned envelope guardian** before actuators.

## Next steps

1. Export to ONNX / TFLite for embedded targets  
2. Harden guardian tests (stall margin, cross-track, wind)  
3. External sim integration (later)
