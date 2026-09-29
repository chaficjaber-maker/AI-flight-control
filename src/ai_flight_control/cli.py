import argparse
import math

from ai_flight_control.agents import PIDAltitudeAgent, PolicyAltitudeAgent
from ai_flight_control.dynamics import LongitudinalState
from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.missions import MissionLibrary, MissionLoadError, load_mission_json
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.sim import run_altitude_hold
from ai_flight_control.stack import FlightControlStack
from ai_flight_control.state import MissionSpec
from ai_flight_control.wind import WindField


def _run_legacy_1d(args: argparse.Namespace) -> None:
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
    print(f"Mode: legacy-1d  agent={args.agent}")
    print(f"Target altitude: {result.target_altitude_m:.1f} m")
    print(f"Final altitude:  {final_alt:.1f} m")
    print(f"Final |error|:   {error:.2f} m")


def _resolve_mission(name: str) -> MissionSpec:
    catalog: dict[str, MissionSpec] = {
        "hold": MissionLibrary.hold_cruise(),
        "climb": MissionLibrary.climb_to_cruise(300, 500),
        "box": MissionLibrary.box_pattern(),
        "training": MissionLibrary.training_route(),
    }
    if name not in catalog:
        raise SystemExit(f"Unknown mission {name!r}. Choose from: {', '.join(catalog)}")
    return catalog[name]


def _resolve_stack_mission(args: argparse.Namespace) -> MissionSpec:
    if args.mission_file:
        try:
            return load_mission_json(args.mission_file)
        except MissionLoadError as exc:
            raise SystemExit(str(exc)) from exc
    return _resolve_mission(args.mission_name)


def _run_stack(args: argparse.Namespace) -> None:
    spec = _resolve_stack_mission(args)
    config = PolicyConfig(
        stability=args.stability,
        maneuver=args.maneuver,
        mission=args.mission_policy,
    )
    stack = FlightControlStack.from_policy_config(
        config,
        train_if_missing=not args.no_train,
    )
    use_wind = args.wind_n != 0.0 or args.wind_e != 0.0 or args.gust > 0.0
    wind = (
        WindField(north_m_s=args.wind_n, east_m_s=args.wind_e, gust_std_m_s=args.gust)
        if use_wind
        else None
    )
    result = stack.run_fixed_wing(
        spec,
        initial=FixedWingState(
            north_m=args.north,
            east_m=args.east,
            altitude_m=args.initial_alt,
            airspeed_m_s=args.airspeed,
            heading_rad=math.radians(args.heading_deg),
            bank_rad=0.0,
            flight_path_angle_rad=0.0,
        ),
        duration_s=args.duration,
        dt=args.dt,
        wind=wind,
    )
    mission_label = args.mission_file or args.mission_name
    print(
        f"Mode: stack  mission={mission_label}  "
        f"policies=(S:{args.stability}, M:{args.maneuver}, Mi:{args.mission_policy})"
    )
    print(
        f"Final N/E/Alt: {result.final_north_m:.1f}, "
        f"{result.final_east_m:.1f}, {result.final_altitude_m:.1f}"
    )
    if result.logs:
        last = result.logs[-1]
        print(f"Final airspeed: {last.airspeed_m_s:.1f} m/s  segment={last.segment_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI flight control simulation CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    legacy = sub.add_parser("legacy", help="1-D altitude hold (original harness)")
    legacy.add_argument("--target", type=float, default=1000.0)
    legacy.add_argument("--initial", type=float, default=800.0)
    legacy.add_argument("--duration", type=float, default=120.0)
    legacy.add_argument("--dt", type=float, default=0.05)
    legacy.add_argument("--agent", choices=("pid", "policy"), default="pid")
    legacy.set_defaults(func=_run_legacy_1d)

    stack = sub.add_parser("stack", help="Three-holder fixed-wing stack")
    stack.add_argument(
        "--mission-file",
        type=str,
        default="",
        help="Path to mission JSON (overrides --mission-name)",
    )
    stack.add_argument(
        "--mission-name",
        choices=("hold", "climb", "box", "training"),
        default="training",
    )
    stack.add_argument("--duration", type=float, default=120.0)
    stack.add_argument("--dt", type=float, default=0.05)
    stack.add_argument("--initial-alt", type=float, default=380.0)
    stack.add_argument("--airspeed", type=float, default=22.0)
    stack.add_argument("--north", type=float, default=0.0)
    stack.add_argument("--east", type=float, default=0.0)
    stack.add_argument("--heading-deg", type=float, default=0.0)
    stack.add_argument("--wind-n", type=float, default=0.0)
    stack.add_argument("--wind-e", type=float, default=0.0)
    stack.add_argument("--gust", type=float, default=0.0)
    stack.add_argument("--stability", choices=("reference", "neural"), default="reference")
    stack.add_argument("--maneuver", choices=("reference", "neural"), default="reference")
    stack.add_argument(
        "--mission-policy",
        dest="mission_policy",
        choices=("reference", "neural"),
        default="reference",
    )
    stack.add_argument("--no-train", action="store_true")
    stack.set_defaults(func=_run_stack)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
