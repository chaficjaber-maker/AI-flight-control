from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_flight_control.policies.guardian import apply_stability_guardian
from ai_flight_control.policies.mlp import MLP
from ai_flight_control.state import (
    ActuatorCommand,
    AdaptationContext,
    EnvelopeLimits,
    FlightState,
    ManeuverSetpoint,
    wrap_angle_rad,
)

STABILITY_INPUT_DIM = 12
STABILITY_OUTPUT_DIM = 3
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "stability.npz"


def stability_features(
    state: FlightState,
    setpoint: ManeuverSetpoint,
) -> np.ndarray:
    h_err = (setpoint.target_altitude_m - state.altitude_m) / 500.0
    v_err = (setpoint.target_airspeed_m_s - state.airspeed_m_s) / 15.0
    hdg_err = wrap_angle_rad(setpoint.target_heading_rad - state.heading_rad) / np.pi
    return np.array(
        [
            h_err,
            v_err,
            hdg_err,
            state.flight_path_angle_rad / 0.3,
            state.bank_rad / 0.6,
            state.airspeed_m_s / 30.0,
            setpoint.target_bank_rad / 0.6,
            state.altitude_m / 5000.0,
            np.sin(state.heading_rad),
            np.cos(state.heading_rad),
            state.wind_n_m_s / 15.0,
            state.wind_e_m_s / 15.0,
        ],
        dtype=float,
    )


@dataclass
class NeuralStabilityPolicy:
    limits: EnvelopeLimits = field(default_factory=EnvelopeLimits)
    network: MLP = field(
        default_factory=lambda: MLP(STABILITY_INPUT_DIM, 24, STABILITY_OUTPUT_DIM, seed=1)
    )
    _online_lr: float = 0.0005
    _last_features: np.ndarray | None = None
    _last_output: np.ndarray | None = None
    _envelope_active: bool = False

    def reset(self) -> None:
        self._last_features = None
        self._last_output = None
        self._envelope_active = False

    def load_weights(self, path: Path | None = None) -> None:
        self.network = MLP.load(path or DEFAULT_WEIGHTS)

    def step(
        self,
        state: FlightState,
        setpoint: ManeuverSetpoint,
        dt: float,
    ) -> ActuatorCommand:
        del dt
        features = stability_features(state, setpoint)
        out = self.network.predict(features)
        self._last_features = features
        self._last_output = out.copy()
        raw = ActuatorCommand(
            throttle=float(1.0 / (1.0 + np.exp(-out[0]))),
            pitch=float(np.tanh(out[1])),
            roll=float(np.tanh(out[2])),
        )
        guarded = apply_stability_guardian(raw, state, self.limits)
        self._envelope_active = guarded.envelope_active
        return guarded.command

    @property
    def envelope_active_last_step(self) -> bool:
        return self._envelope_active

    def adapt(self, context: AdaptationContext) -> None:
        """One-step online correction toward lower tracking error (bounded)."""
        if self._last_features is None or self._last_output is None:
            return
        target_delta = np.array(
            [
                -0.1 * context.altitude_error_m / 500.0,
                0.08 * context.altitude_error_m / 500.0 - 0.05 * context.airspeed_error_m_s / 15.0,
                0.15 * context.heading_error_rad / np.pi,
            ],
            dtype=float,
        )
        if context.envelope_active:
            target_delta *= 0.3
        desired = self._last_output - target_delta
        err = self._last_output - desired
        h = self.network._last_h
        if h is None:
            return
        grad_y = err
        grad_h = grad_y @ self.network.weights.w2.T * (1.0 - h * h)
        self.network.weights.w2 -= self._online_lr * np.outer(h, grad_y)
        self.network.weights.w1 -= self._online_lr * np.outer(
            self._last_features, grad_h
        )
