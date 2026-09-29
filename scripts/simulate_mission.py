#!/usr/bin/env python3
"""Run a full mission + three-layer flight control simulation."""

import argparse
from pathlib import Path

from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.simulate import (
    MissionSimulationConfig,
    print_simulation_summary,
    run_mission_simulation,
    write_simulation_csv,
    write_simulation_report_json,
)
from ai_flight_control.wind import WindField


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate mission with flight control stack.")
    parser.add_argument(
        "--mission-file",
        type=str,
        default="missions/examples/box.json",
    )
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--initial-alt", type=float, default=380.0)
    parser.add_argument("--airspeed", type=float, default=22.0)
    parser.add_argument("--wind-n", type=float, default=3.0)
    parser.add_argument("--wind-e", type=float, default=-2.0)
    parser.add_argument("--gust", type=float, default=0.0)
    parser.add_argument("--stability", choices=("reference", "neural"), default="reference")
    parser.add_argument("--maneuver", choices=("reference", "neural"), default="reference")
    parser.add_argument("--mission-policy", choices=("reference", "neural"), default="reference")
    parser.add_argument("--all-neural", action="store_true")
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--csv", type=str, default="simulation_logs/flight.csv")
    parser.add_argument("--report", type=str, default="simulation_logs/report.json")
    args = parser.parse_args()

    policy = PolicyConfig.all_neural() if args.all_neural else PolicyConfig(
        stability=args.stability,
        maneuver=args.maneuver,
        mission=args.mission_policy,
    )
    wind = None if args.no_wind else WindField(args.wind_n, args.wind_e, args.gust)

    config = MissionSimulationConfig(
        mission_file=args.mission_file,
        policy=policy,
        duration_s=args.duration,
        dt=args.dt,
        initial=FixedWingState(
            0, 0, args.initial_alt, args.airspeed, 0, 0, 0
        ),
        wind=wind,
        train_if_missing=not args.no_train,
    )

    result, report = run_mission_simulation(config)
    print_simulation_summary(report)
    write_simulation_csv(result, Path(args.csv))
    write_simulation_report_json(report, Path(args.report))
    print(f"Wrote CSV log: {args.csv}")
    print(f"Wrote report:  {args.report}")


if __name__ == "__main__":
    main()
