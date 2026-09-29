import argparse

from ai_flight_control.agents import PIDAltitudeAgent, PolicyAltitudeAgent
from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.sim import run_altitude_hold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 1-D altitude-hold simulation.")
    parser.add_argument("--target", type=float, default=1000.0, help="Target altitude (m)")
    parser.add_argument("--initial", type=float, default=800.0, help="Initial altitude (m)")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation length (s)")
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep (s)")
    parser.add_argument(
        "--agent",
        choices=("pid", "policy"),
        default="pid",
        help="Control agent (policy uses PID fallback until ML is added)",
    )
    args = parser.parse_args()

    agent = PIDAltitudeAgent() if args.agent == "pid" else PolicyAltitudeAgent()
    result = run_altitude_hold(
        agent,
        initial=LongitudinalState(args.initial, 0.0),
        target_altitude_m=args.target,
        duration_s=args.duration,
        dt=args.dt,
    )

    final_alt = result.altitudes_m[-1]
    error = abs(final_alt - result.target_altitude_m)
    print(f"Agent: {args.agent}")
    print(f"Target altitude: {result.target_altitude_m:.1f} m")
    print(f"Final altitude:  {final_alt:.1f} m")
    print(f"Final |error|:   {error:.2f} m")


if __name__ == "__main__":
    main()
