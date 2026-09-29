from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_flight_control.state import (
    MissionSegment,
    MissionSegmentKind,
    MissionSpec,
    WaypointTarget,
)

_SEGMENT_KINDS = {k.value for k in MissionSegmentKind}


class MissionLoadError(ValueError):
    """Invalid mission JSON."""


def _require_mapping(obj: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise MissionLoadError(f"{context} must be a JSON object")
    return obj


def _parse_waypoint(obj: dict[str, Any], *, context: str) -> WaypointTarget:
    try:
        return WaypointTarget(
            north_m=float(obj["north_m"]),
            east_m=float(obj["east_m"]),
            altitude_m=float(obj["altitude_m"]),
            airspeed_m_s=float(obj.get("airspeed_m_s", 22.0)),
            capture_radius_m=float(obj.get("capture_radius_m", 40.0)),
        )
    except KeyError as exc:
        raise MissionLoadError(f"{context} missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise MissionLoadError(f"{context} has invalid numeric field") from exc


def _parse_segment(obj: Any, index: int) -> MissionSegment:
    ctx = f"segments[{index}]"
    data = _require_mapping(obj, context=ctx)
    kind_raw = data.get("kind")
    if kind_raw not in _SEGMENT_KINDS:
        raise MissionLoadError(
            f"{ctx}.kind must be one of {sorted(_SEGMENT_KINDS)}, got {kind_raw!r}"
        )
    kind = MissionSegmentKind(kind_raw)

    if kind == MissionSegmentKind.FLY_TO_WAYPOINT:
        wp = _parse_waypoint(data, context=ctx)
        return MissionSegment.fly_to(wp)

    if "altitude_m" not in data:
        raise MissionLoadError(f"{ctx} requires altitude_m for kind {kind_raw!r}")

    return MissionSegment(
        kind=kind,
        target_altitude_m=float(data["altitude_m"]),
        max_vertical_rate_m_s=float(data.get("max_vertical_rate_m_s", 5.0)),
    )


def mission_from_dict(data: dict[str, Any]) -> MissionSpec:
    """Parse a mission object into MissionSpec."""
    root = _require_mapping(data, context="root")

    if "segments" in root:
        segments_raw = root["segments"]
        if not isinstance(segments_raw, list) or not segments_raw:
            raise MissionLoadError("segments must be a non-empty array")
        segments = [_parse_segment(item, i) for i, item in enumerate(segments_raw)]
        return MissionSpec(segments=segments)

    if "waypoints" in root:
        waypoints_raw = root["waypoints"]
        if not isinstance(waypoints_raw, list) or not waypoints_raw:
            raise MissionLoadError("waypoints must be a non-empty array")
        waypoints = [
            _parse_waypoint(_require_mapping(wp, context=f"waypoints[{i}]"), context=f"waypoints[{i}]")
            for i, wp in enumerate(waypoints_raw)
        ]
        return MissionSpec.waypoint_route(waypoints)

    raise MissionLoadError('mission JSON must contain "segments" or "waypoints"')


def load_mission_json(path: str | Path) -> MissionSpec:
    """Load MissionSpec from a JSON file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise MissionLoadError(f"mission file not found: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissionLoadError(f"invalid JSON in {file_path}: {exc}") from exc
    return mission_from_dict(data)


def mission_to_dict(spec: MissionSpec) -> dict[str, Any]:
    """Serialize MissionSpec to a JSON-friendly dict."""
    segments: list[dict[str, Any]] = []
    for seg in spec.segments:
        if seg.kind == MissionSegmentKind.FLY_TO_WAYPOINT and seg.waypoint is not None:
            wp = seg.waypoint
            segments.append(
                {
                    "kind": seg.kind.value,
                    "north_m": wp.north_m,
                    "east_m": wp.east_m,
                    "altitude_m": wp.altitude_m,
                    "airspeed_m_s": wp.airspeed_m_s,
                    "capture_radius_m": wp.capture_radius_m,
                }
            )
        else:
            segments.append(
                {
                    "kind": seg.kind.value,
                    "altitude_m": seg.target_altitude_m,
                    "max_vertical_rate_m_s": seg.max_vertical_rate_m_s,
                }
            )
    return {"segments": segments}


def save_mission_json(spec: MissionSpec, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(mission_to_dict(spec), indent=2) + "\n", encoding="utf-8")
