from dataclasses import dataclass, field

from ai_flight_control.dynamics import LongitudinalParams, LongitudinalState, step_longitudinal
from ai_flight_control.fixed_wing_plant import (
    FixedWingControls,
    FixedWingParams,
    FixedWingState,
    step_fixed_wing,
)
from ai_flight_control.layers.maneuver import ManeuverLayer
from ai_flight_control.layers.mission import MissionLayer
from ai_flight_control.layers.stability import StabilityLayer
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    FlightState,
    MissionSpec,
    wrap_angle_rad,
)


@dataclass
class StackStepLog:
    time_s: float
    altitude_m: float
    vertical_velocity_m_s: float
    thrust: float
    maneuver_label: str
    segment_index: int
    airspeed_m_s: float = 0.0
    north_m: float = 0.0
    east_m: float = 0.0
    heading_deg: float = 0.0


@dataclass
class StackRunResult:
    logs: list[StackStepLog]
    final_altitude_m: float
    mission_spec: MissionSpec
    final_north_m: float = 0.0
    final_east_m: float = 0.0


@dataclass
class FlightControlStack:
    """
    Composes mission, maneuver, and stability layers (all adaptive policies).
    """

    mission: MissionLayer = field(default_factory=MissionLayer)
    maneuver: ManeuverLayer = field(default_factory=ManeuverLayer)
    stability: StabilityLayer = field(default_factory=StabilityLayer)
    adapt_every_n_steps: int = 20

    def reset(self, spec: MissionSpec, *, legacy_vertical_only: bool = False) -> None:
        self.mission.reset(spec)
        self.maneuver.reset()
        self.stability.legacy_vertical_only = legacy_vertical_only
        self.stability.reset()

    def step(
        self,
        state: FlightState,
        *,
        dt: float,
        step_index: int,
    ) -> tuple[ActuatorCommand, StackStepLog]:
        intent = self.mission.step(state)
        maneuver_sp = self.maneuver.step(state, intent)
        command = self.stability.step(state, maneuver_sp, dt)

        if step_index > 0 and step_index % self.adapt_every_n_steps == 0:
            ctx = AdaptationContext(
                altitude_error_m=maneuver_sp.target_altitude_m - state.altitude_m,
                airspeed_error_m_s=maneuver_sp.target_airspeed_m_s - state.airspeed_m_s,
                heading_error_rad=wrap_angle_rad(
                    maneuver_sp.target_heading_rad - state.heading_rad
                ),
                vertical_rate_error_m_s=(
                    maneuver_sp.target_vertical_rate_m_s - state.vertical_velocity_m_s
                ),
                cross_track_error_m=intent.distance_to_target_m,
                envelope_active=self.stability.envelope_active_last_step,
                segment_index=intent.segment_index,
            )
            self.mission.adapt(ctx)
            self.maneuver.adapt(ctx)
            self.stability.adapt(ctx)

        log = StackStepLog(
            time_s=state.time_s,
            altitude_m=state.altitude_m,
            vertical_velocity_m_s=state.vertical_velocity_m_s,
            thrust=command.normalized_thrust,
            maneuver_label=maneuver_sp.maneuver_label,
            segment_index=intent.segment_index,
            airspeed_m_s=state.airspeed_m_s,
            north_m=state.north_m,
            east_m=state.east_m,
            heading_deg=float(state.heading_rad * 57.2958),
        )
        return command, log

    @staticmethod
    def _flight_state_from_longitudinal(
        plant: LongitudinalState, *, time_s: float, cruise_airspeed_m_s: float
    ) -> FlightState:
        v = cruise_airspeed_m_s
        gamma = 0.0
        if v > 1.0:
            gamma = max(-0.35, min(0.35, plant.vertical_velocity_m_s / v))
        return FlightState(
            altitude_m=plant.altitude_m,
            airspeed_m_s=v,
            flight_path_angle_rad=gamma,
            time_s=time_s,
        )

    def run(
        self,
        spec: MissionSpec,
        *,
        initial: LongitudinalState,
        duration_s: float,
        dt: float,
        params: LongitudinalParams | None = None,
        cruise_airspeed_m_s: float = 22.0,
    ) -> StackRunResult:
        if duration_s <= 0 or dt <= 0:
            raise ValueError("duration_s and dt must be positive")

        self.reset(spec, legacy_vertical_only=True)
        plant = initial
        logs: list[StackStepLog] = []
        t = 0.0
        steps = int(duration_s / dt)
        for i in range(steps + 1):
            fstate = self._flight_state_from_longitudinal(
                plant, time_s=t, cruise_airspeed_m_s=cruise_airspeed_m_s
            )
            cmd, log = self.step(fstate, dt=dt, step_index=i)
            logs.append(log)
            if t >= duration_s:
                break
            plant = step_longitudinal(plant, cmd.normalized_thrust, dt, params)
            t += dt

        return StackRunResult(
            logs=logs,
            final_altitude_m=plant.altitude_m,
            mission_spec=spec,
        )

    def run_fixed_wing(
        self,
        spec: MissionSpec,
        *,
        initial: FixedWingState,
        duration_s: float,
        dt: float,
        params: FixedWingParams | None = None,
    ) -> StackRunResult:
        if duration_s <= 0 or dt <= 0:
            raise ValueError("duration_s and dt must be positive")

        self.reset(spec, legacy_vertical_only=False)
        plant = initial
        logs: list[StackStepLog] = []
        t = 0.0
        steps = int(duration_s / dt)
        for i in range(steps + 1):
            fstate = FlightState(
                altitude_m=plant.altitude_m,
                airspeed_m_s=plant.airspeed_m_s,
                north_m=plant.north_m,
                east_m=plant.east_m,
                heading_rad=plant.heading_rad,
                bank_rad=plant.bank_rad,
                flight_path_angle_rad=plant.flight_path_angle_rad,
                time_s=t,
            )
            cmd, log = self.step(fstate, dt=dt, step_index=i)
            logs.append(log)
            if t >= duration_s:
                break
            controls = FixedWingControls(
                throttle=cmd.throttle,
                pitch=cmd.pitch,
                roll=cmd.roll,
            )
            plant = step_fixed_wing(plant, controls, dt, params)
            t += dt

        return StackRunResult(
            logs=logs,
            final_altitude_m=plant.altitude_m,
            final_north_m=plant.north_m,
            final_east_m=plant.east_m,
            mission_spec=spec,
        )
