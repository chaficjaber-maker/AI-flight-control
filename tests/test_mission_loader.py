import json
from pathlib import Path

import pytest

from ai_flight_control.missions import (
    MissionLibrary,
    MissionLoadError,
    load_mission_json,
    mission_from_dict,
    mission_to_dict,
    save_mission_json,
)
from ai_flight_control.state import MissionSegmentKind


def test_load_segments_json(tmp_path: Path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"kind": "climb_to", "altitude_m": 500},
                    {
                        "kind": "fly_to_waypoint",
                        "north_m": 700,
                        "east_m": 50,
                        "altitude_m": 500,
                        "capture_radius_m": 100,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = load_mission_json(path)
    assert len(spec.segments) == 2
    assert spec.segments[0].kind == MissionSegmentKind.CLIMB_TO
    assert spec.segments[1].waypoint is not None
    assert spec.segments[1].waypoint.north_m == 700


def test_load_waypoints_shorthand(tmp_path: Path):
    path = tmp_path / "wp.json"
    path.write_text(
        json.dumps(
            {
                "waypoints": [
                    {"north_m": 1, "east_m": 2, "altitude_m": 400},
                    {"north_m": 3, "east_m": 4, "altitude_m": 420},
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = load_mission_json(path)
    assert len(spec.segments) == 2
    assert all(s.kind == MissionSegmentKind.FLY_TO_WAYPOINT for s in spec.segments)


def test_roundtrip_library_mission(tmp_path: Path):
    spec = MissionLibrary.training_route()
    out = tmp_path / "out.json"
    save_mission_json(spec, out)
    loaded = load_mission_json(out)
    assert len(loaded.segments) == len(spec.segments)
    assert mission_to_dict(loaded) == mission_to_dict(spec)


def test_invalid_kind_raises():
    with pytest.raises(MissionLoadError):
        mission_from_dict({"segments": [{"kind": "orbit", "altitude_m": 100}]})


def test_repo_example_box_json():
    path = Path(__file__).resolve().parents[1] / "missions" / "examples" / "box.json"
    spec = load_mission_json(path)
    assert len(spec.segments) >= 4
