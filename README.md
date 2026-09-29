# AI-flight-control

Simulation and control stack for experimenting with **AI-assisted flight control**, starting from a small runnable baseline you can extend toward learned policies, full-state dynamics, or simulator integration (JSBSim, X-Plane, MAVLink, etc.).

## What’s included

- **1-D altitude plant** — simplified vertical dynamics for fast iteration
- **PID altitude-hold autopilot** — classical baseline controller
- **`PolicyAltitudeAgent`** — same interface as PID; ready to swap in ML/RL weights later
- **CLI** — quick end-to-end runs from the terminal

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
flight-sim --target 1000 --initial 800 --duration 120
```

## Project layout

```
src/ai_flight_control/
  dynamics.py      # plant model
  pid.py           # reusable PID
  sim.py           # simulation loop
  agents/          # FlightAgent implementations
tests/
```

## Roadmap (suggested)

1. **Lateral–directional channel** or full 6-DOF linearized model  
2. **Observation/action spaces** for RL (Gymnasium env wrapper)  
3. **Training pipeline** (e.g. PPO on altitude + airspeed tracking)  
4. **Hardware-in-the-loop** via MAVLink / PX4 SITL  

Tell us which direction you want first (research sim, RL training, or autopilot integration), and we can prioritize the next slice of work.
