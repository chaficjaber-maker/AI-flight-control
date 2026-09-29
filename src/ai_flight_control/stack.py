from dataclasses import dataclass, field

from ai_flight_control.dynamics import LongitudinalParams, LongitudinalState, step_longitudinal
from ai_flight_control.layers.maneuver import ManeuverLayer
from ai_flight_control.layers.mission import MissionLayer
from ai_flight_control.layers.stability import StabilityLayer
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    FlightState,
    MissionSpec,
)


@dataclass
class StackStepLog:
    time_s: float
    altitude_m: float
    vertical_velocity_m_s: float
    thrust: float
    maneuver_label: str
    segment_index: int


@dataclass
class StackRunResult:
    logs: list[StackStepLog]
    final_altitude_m: float
    mission_spec: MissionSpec


@dataclass
class FlightControlStack:
    """
    Composes mission, maneuver, and stability layers (all adaptive policies).
    """

    mission: MissionLayer = field(default_factory=MissionLayer)
    maneuver: ManeuverLayer = field(default_factory=ManeuverLayer)
    stability: StabilityLayer = field(default_factory=StabilityLayer)
    adapt_every_n_steps: int = 20

    def reset(self, spec: MissionSpec) -> None:
        self.mission.reset(spec)
        self.maneuver.reset()
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
                vertical_rate_error_m_s=(
                    maneuver_sp.target_vertical_rate_m_s - state.vertical_velocity_m_s
                ),
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
        )
        return command, log

    def run(
        self,
        spec: MissionSpec,
        *,
        initial: LongitudinalState,
        duration_s: float,
        dt: float,
        params: LongitudinalParams | None = None,
    ) -> StackRunResult:
        if duration_s <= 0 or dt <= 0:
            raise ValueError("duration_s and dt must be positive")

        self.reset(spec)
        plant = initial
        logs: list[StackStepLog] = []
        t = 0.0
        steps = int(duration_s / dt)
        for i in range(steps + 1):
            fstate = FlightState(
                altitude_m=plant.altitude_m,
                vertical_velocity_m_s=plant.vertical_velocity_m_s,
                time_s=t,
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
