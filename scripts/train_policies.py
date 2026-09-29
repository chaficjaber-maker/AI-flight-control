#!/usr/bin/env python3
"""Train stability, maneuver, and mission MLP policies (in that order)."""

from ai_flight_control.policies.train import train_all_in_order


def main() -> None:
    paths = train_all_in_order(quick=False)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
