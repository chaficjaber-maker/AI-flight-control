"""Rich behavioral-cloning datasets (rollouts + synthetic coverage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_flight_control.fixed_wing_plant import FixedWingControls, FixedWingState, step_fixed_wing
from ai_flight_control.layers.maneuver import ManeuverLayer
from ai_flight_control.layers.mission import MissionLayer
from ai_flight_control.layers.stability import StabilityLayer
from ai_flight_control.policies.maneuver_policy import maneuver_features
from ai_flight_control.policies.mission_policy import mission_features
from ai_flight_control.policies.stability_policy import stability_features
from ai_flight_control.state import (
    FlightState,
    MissionIntent,
    MissionSegment,
    MissionSegmentKind,
    MissionSpec,
    ManeuverSetpoint,
    WaypointTarget,
)
from ai_flight_control.wind import WindField


@dataclass(frozen=True)
class DatasetConfig:
    stability_samples: int = 2500
    maneuver_samples: int = 2500
    mission_samples: int = 1800
    rollout_steps: int = 40
    seed: int = 42


def _random_wind(rng: np.random.Generator) -> WindField:
    roll = rng.random()
    if roll < 0.5:
        return WindField.calm()
    if roll < 0.85:
        speed = float(rng.uniform(2.0, 10.0))
        angle = float(rng.uniform(-np.pi, np.pi))
        return WindField(
            north_m_s=speed * np.cos(angle),
            east_m_s=speed * np.sin(angle),
        )
    return WindField(
        north_m_s=float(rng.uniform(-6.0, 6.0)),
        east_m_s=float(rng.uniform(-6.0, 6.0)),
        gust_std_m_s=float(rng.uniform(0.5, 2.0)),
    )


def _state_from_plant(plant: FixedWingState, *, wind_n: float, wind_e: float, t: float) -> FlightState:
    return FlightState(
        altitude_m=plant.altitude_m,
        airspeed_m_s=plant.airspeed_m_s,
        north_m=plant.north_m,
        east_m=plant.east_m,
        heading_rad=plant.heading_rad,
        bank_rad=plant.bank_rad,
        flight_path_angle_rad=plant.flight_path_angle_rad,
        wind_n_m_s=wind_n,
        wind_e_m_s=wind_e,
        time_s=t,
    )


def collect_stability_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    teacher = StabilityLayer()
    teacher.legacy_vertical_only = False
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    n_synthetic = config.stability_samples // 2
    n_rollout = config.stability_samples - n_synthetic

    for _ in range(n_synthetic):
        wind = _random_wind(rng)
        wn, we = wind.north_m_s, wind.east_m_s
        state = FlightState(
            altitude_m=float(rng.uniform(150, 1400)),
            airspeed_m_s=float(rng.uniform(15, 30)),
            north_m=float(rng.uniform(0, 2000)),
            east_m=float(rng.uniform(-600, 600)),
            heading_rad=float(rng.uniform(-np.pi, np.pi)),
            bank_rad=float(rng.uniform(-0.45, 0.45)),
            flight_path_angle_rad=float(rng.uniform(-0.15, 0.15)),
            wind_n_m_s=wn,
            wind_e_m_s=we,
        )
        sp = ManeuverSetpoint(
            target_altitude_m=float(rng.uniform(150, 1400)),
            target_airspeed_m_s=float(rng.uniform(17, 28)),
            target_heading_rad=state.heading_rad + float(rng.uniform(-1.0, 1.0)),
            target_bank_rad=float(rng.uniform(-0.4, 0.4)),
            maneuver_label="track",
        )
        teacher.reset()
        cmd = teacher.step(state, sp, dt=0.05)
        xs.append(stability_features(state, sp))
        ys.append(_encode_stability(cmd.throttle, cmd.pitch, cmd.roll))

    rollout_teacher = StabilityLayer()
    rollout_teacher.legacy_vertical_only = False
    specs = [
        MissionSpec.climb_then_hold(250, 600),
        MissionSpec.waypoint_route(
            [
                WaypointTarget(400, -100, 450, 22),
                WaypointTarget(900, 150, 500, 24),
            ]
        ),
        MissionSpec.hold_at(350),
    ]
    per_spec = max(1, n_rollout // len(specs))
    for spec in specs:
        wind = _random_wind(rng)
        plant = FixedWingState(0, 0, 350, 22, 0, 0, 0)
        mission = MissionLayer()
        maneuver = ManeuverLayer()
        mission.reset(spec)
        maneuver.reset()
        rollout_teacher.reset()
        wn, we = wind.north_m_s, wind.east_m_s
        for step in range(per_spec):
            t = step * 0.05
            if wind.gust_std_m_s > 0:
                wn, we = wind.sample(rng)
            fstate = _state_from_plant(plant, wind_n=wn, wind_e=we, t=t)
            intent = mission.step(fstate)
            sp = maneuver.step(fstate, intent)
            cmd = rollout_teacher.step(fstate, sp, dt=0.05)
            xs.append(stability_features(fstate, sp))
            ys.append(_encode_stability(cmd.throttle, cmd.pitch, cmd.roll))
            plant = step_fixed_wing(
                plant,
                FixedWingControls(cmd.throttle, cmd.pitch, cmd.roll),
                0.05,
                wind=WindField(wn, we),
            )

    return np.stack(xs), np.stack(ys)


def _encode_stability(throttle: float, pitch: float, roll: float) -> np.ndarray:
    t = float(np.clip(throttle, 1e-4, 1.0 - 1e-4))
    return np.array(
        [
            float(np.log(t / (1.0 - t))),
            float(np.arctanh(np.clip(pitch, -0.999, 0.999))),
            float(np.arctanh(np.clip(roll, -0.999, 0.999))),
        ]
    )


def collect_maneuver_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 1)
    teacher = ManeuverLayer()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    scenarios: list[tuple[FlightState, MissionIntent]] = []
    for _ in range(config.maneuver_samples):
        wind = _random_wind(rng)
        kind_roll = rng.random()
        if kind_roll < 0.35:
            seg = MissionSegment(
                MissionSegmentKind.CLIMB_TO,
                target_altitude_m=float(rng.uniform(400, 900)),
            )
        elif kind_roll < 0.55:
            seg = MissionSegment(
                MissionSegmentKind.DESCEND_TO,
                target_altitude_m=float(rng.uniform(200, 400)),
            )
        elif kind_roll < 0.75:
            seg = MissionSegment(
                MissionSegmentKind.HOLD_ALTITUDE,
                target_altitude_m=float(rng.uniform(300, 700)),
            )
        else:
            wp = WaypointTarget(
                north_m=float(rng.uniform(100, 2500)),
                east_m=float(rng.uniform(-800, 800)),
                altitude_m=float(rng.uniform(280, 850)),
                airspeed_m_s=float(rng.uniform(19, 26)),
            )
            seg = MissionSegment.fly_to(wp)

        state = FlightState(
            altitude_m=float(rng.uniform(200, 800)),
            airspeed_m_s=float(rng.uniform(16, 28)),
            north_m=float(rng.uniform(0, 1200)),
            east_m=float(rng.uniform(-500, 500)),
            heading_rad=float(rng.uniform(-np.pi, np.pi)),
            bank_rad=float(rng.uniform(-0.3, 0.3)),
            flight_path_angle_rad=float(rng.uniform(-0.1, 0.1)),
            wind_n_m_s=wind.north_m_s,
            wind_e_m_s=wind.east_m_s,
        )
        intent = MissionIntent(
            segment_index=int(rng.integers(0, 3)),
            segment=seg,
            segment_complete=False,
            bearing_to_target_rad=float(rng.uniform(-np.pi, np.pi)),
            distance_to_target_m=float(rng.uniform(30, 2000)),
        )
        scenarios.append((state, intent))

    for state, intent in scenarios:
        teacher.reset()
        sp = teacher.step(state, intent)
        xs.append(maneuver_features(state, intent))
        wp = intent.segment.waypoint
        base_alt = wp.altitude_m if wp else intent.segment.target_altitude_m
        base_spd = wp.airspeed_m_s if wp else 22.0
        ys.append(
            np.array(
                [
                    (sp.target_altitude_m - base_alt) / 50.0,
                    (sp.target_airspeed_m_s - base_spd) / 3.0,
                    0.0,
                    sp.target_bank_rad / np.deg2rad(25.0),
                    sp.target_vertical_rate_m_s / 5.0,
                    0.0,
                ]
            )
        )

    return np.stack(xs), np.stack(ys)


def collect_mission_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 2)
    teacher = MissionLayer()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    route_specs = [
        MissionSpec.waypoint_route(
            [
                WaypointTarget(300, 0, 400, 22, 60),
                WaypointTarget(700, -120, 430, 23, 70),
                WaypointTarget(1100, 200, 480, 22, 80),
            ]
        ),
        MissionSpec(
            segments=[
                MissionSegment(MissionSegmentKind.CLIMB_TO, 550),
                MissionSegment(MissionSegmentKind.HOLD_ALTITUDE, 550),
                MissionSegment.fly_to(WaypointTarget(600, 300, 550, 22, 90)),
            ]
        ),
    ]

    per = max(1, config.mission_samples // len(route_specs))
    for spec in route_specs:
        teacher.reset(spec)
        plant = FixedWingState(0, 0, 380, 22, 0.2, 0, 0)
        wind = _random_wind(rng)
        for step in range(per):
            wn, we = wind.north_m_s, wind.east_m_s
            if wind.gust_std_m_s > 0:
                wn, we = wind.sample(rng)
            state = _state_from_plant(plant, wind_n=wn, wind_e=we, t=step * 0.05)
            intent = teacher.step(state)
            feat = mission_features(state, intent)
            wp = intent.segment.waypoint
            alt_tgt = wp.altitude_m if wp else intent.segment.target_altitude_m
            alt_err = abs(state.altitude_m - alt_tgt)
            tol_out = float(np.tanh((alt_err - 15.0) / 25.0))
            cap_out = float(np.tanh((intent.distance_to_target_m - 80.0) / 120.0))
            xs.append(feat)
            ys.append(np.array([tol_out, cap_out]))
            plant = step_fixed_wing(
                plant,
                FixedWingControls(0.65, 0.05, 0.1),
                0.05,
                wind=WindField(wn, we),
            )

    return np.stack(xs), np.stack(ys)
