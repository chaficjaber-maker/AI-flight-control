#!/usr/bin/env python3
"""Export trained MLP policies to JSON + C headers for embedded targets."""

from ai_flight_control.policies.export_embedded import export_all_from_npz


def main() -> None:
    manifest = export_all_from_npz()
    print(manifest.to_json())


if __name__ == "__main__":
    main()
