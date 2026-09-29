from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai_flight_control.fixed_wing_plant import FixedWingState
from ai_flight_control.missions import load_mission_json
from ai_flight_control.policy_config import PolicyConfig
from ai_flight_control.stack import FlightControlStack, StackRunResult
from ai_flight_control.state import MissionSpec
from ai_flight_control.wind import WindField


@dataclass
class MissionSimulationConfig:
    mission: MissionSpec | None = None
    mission_file: str | Path | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig.default_reference)
    duration_s: float = 180.0
    dt: float = 0.05
    initial: FixedWingState = field(
        default_factory=lambda: FixedWingState(0, 0, 380, 22, 0, 0, 0)
    )
    wind: WindField | None = None
    train_if_missing: bool = True


@dataclass
class SimulationReport:
    mission_segments: int
    duration_s: float
    steps: int
    final_north_m: float
    final_east_m: float
    final_altitude_m: float
    final_airspeed_m_s: float
    max_segment_index: int
    policy_stability: str
    policy_maneuver: str
    policy_mission: str
    max_altitude_error_m: float
    mean_throttle: float
    maneuvers: dict[str, int]

    @staticmethod
    def from_result(
        result: StackRunResult,
        config: MissionSimulationConfig,
    ) -> "SimulationReport":
        logs = result.logs
        if not logs:
            raise ValueError("empty simulation log")

        alt_errors: list[float] = []
        for log in logs:
            seg_idx = min(log.segment_index, len(result.mission_spec.segments) - 1)
            seg = result.mission_spec.segments[seg_idx]
            tgt = seg.target_altitude_m
            if seg.waypoint is not None:
                tgt = seg.waypoint.altitude_m
            alt_errors.append(abs(log.altitude_m - tgt))

        maneuver_counts: dict[str, int] = {}
        for log in logs:
            maneuver_counts[log.maneuver_label] = maneuver_counts.get(log.maneuver_label, 0) + 1

        last = logs[-1]
        return SimulationReport(
            mission_segments=len(result.mission_spec.segments),
            duration_s=config.duration_s,
            steps=len(logs),
            final_north_m=result.final_north_m,
            final_east_m=result.final_east_m,
            final_altitude_m=result.final_altitude_m,
            final_airspeed_m_s=last.airspeed_m_s,
            max_segment_index=max(log.segment_index for log in logs),
            policy_stability=config.policy.stability,
            policy_maneuver=config.policy.maneuver,
            policy_mission=config.policy.mission,
            max_altitude_error_m=max(alt_errors) if alt_errors else 0.0,
            mean_throttle=sum(log.thrust for log in logs) / len(logs),
            maneuvers=maneuver_counts,
        )


def _resolve_mission(config: MissionSimulationConfig) -> MissionSpec:
    if config.mission is not None:
        return config.mission
    if config.mission_file is not None:
        return load_mission_json(config.mission_file)
    raise ValueError("MissionSimulationConfig requires mission or mission_file")


def run_mission_simulation(config: MissionSimulationConfig) -> tuple[StackRunResult, SimulationReport]:
    spec = _resolve_mission(config)
    stack = FlightControlStack.from_policy_config(
        config.policy,
        train_if_missing=config.train_if_missing,
    )
    result = stack.run_fixed_wing(
        spec,
        initial=config.initial,
        duration_s=config.duration_s,
        dt=config.dt,
        wind=config.wind,
    )
    report = SimulationReport.from_result(result, config)
    return result, report


def write_simulation_csv(result: StackRunResult, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "north_m",
                "east_m",
                "altitude_m",
                "airspeed_m_s",
                "heading_deg",
                "segment_index",
                "maneuver",
                "thrust_norm",
            ]
        )
        for log in result.logs:
            writer.writerow(
                [
                    f"{log.time_s:.2f}",
                    f"{log.north_m:.2f}",
                    f"{log.east_m:.2f}",
                    f"{log.altitude_m:.2f}",
                    f"{log.airspeed_m_s:.2f}",
                    f"{log.heading_deg:.2f}",
                    log.segment_index,
                    log.maneuver_label,
                    f"{log.thrust:.3f}",
                ]
            )


def write_simulation_report_json(report: SimulationReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def print_simulation_summary(report: SimulationReport) -> None:
    path_len = math.hypot(report.final_north_m, report.final_east_m)
    print("=== Mission + flight control simulation ===")
    print(
        f"Policies: stability={report.policy_stability}  "
        f"maneuver={report.policy_maneuver}  mission={report.policy_mission}"
    )
    print(
        f"Segments: {report.mission_segments}  "
        f"max segment reached: {report.max_segment_index}  "
        f"steps: {report.steps}  T={report.duration_s:.0f}s"
    )
    print(
        f"Final state: N={report.final_north_m:.1f}m  E={report.final_east_m:.1f}m  "
        f"Alt={report.final_altitude_m:.1f}m  IAS={report.final_airspeed_m_s:.1f}m/s  "
        f"path≈{path_len:.0f}m"
    )
    print(f"Max |altitude error|: {report.max_altitude_error_m:.1f} m")
    print(f"Maneuver counts: {report.maneuvers}")
