"""Export NumPy MLP weights for embedded C/C++ inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_flight_control.policies.mlp import MLP

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
EMBEDDED_DIR = WEIGHTS_DIR / "embedded"


@dataclass(frozen=True)
class EmbeddedManifest:
    models: dict[str, dict[str, int | list[int]]]

    def to_json(self) -> str:
        return json.dumps({"models": self.models}, indent=2)


def mlp_to_dict(net: MLP, name: str) -> dict:
    return {
        "name": name,
        "input_dim": net.input_dim,
        "hidden_dim": net.hidden_dim,
        "output_dim": net.output_dim,
        "w1": net.weights.w1.flatten().tolist(),
        "b1": net.weights.b1.flatten().tolist(),
        "w2": net.weights.w2.flatten().tolist(),
        "b2": net.weights.b2.flatten().tolist(),
        "w1_shape": list(net.weights.w1.shape),
        "w2_shape": list(net.weights.w2.shape),
    }


def export_mlp_json(net: MLP, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(mlp_to_dict(net, name), indent=2), encoding="utf-8")
    return path


def export_mlp_c_header(net: MLP, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name.replace("-", "_")
    w1 = net.weights.w1.flatten()
    b1 = net.weights.b1.flatten()
    w2 = net.weights.w2.flatten()
    b2 = net.weights.b2.flatten()

    def fmt(arr: np.ndarray, per_line: int = 8) -> str:
        vals = ", ".join(f"{x:.8f}f" for x in arr)
        return vals

    content = f"""/* Auto-generated flight policy weights: {name} */
#pragma once
#include <math.h>

#define {safe.upper()}_INPUT_DIM {net.input_dim}
#define {safe.upper()}_HIDDEN_DIM {net.hidden_dim}
#define {safe.upper()}_OUTPUT_DIM {net.output_dim}

static const float {safe}_w1[{len(w1)}] = {{ {fmt(w1)} }};
static const float {safe}_b1[{len(b1)}] = {{ {fmt(b1)} }};
static const float {safe}_w2[{len(w2)}] = {{ {fmt(w2)} }};
static const float {safe}_b2[{len(b2)}] = {{ {fmt(b2)} }};

static inline float {safe}_tanh_f(float x) {{
    return tanhf(x);
}}

void {safe}_forward(const float *input, float *output) {{
    float hidden[{net.hidden_dim}];
    for (int j = 0; j < {net.hidden_dim}; j++) {{
        float s = {safe}_b1[j];
        for (int i = 0; i < {net.input_dim}; i++) {{
            s += input[i] * {safe}_w1[i * {net.hidden_dim} + j];
        }}
        hidden[j] = {safe}_tanh_f(s);
    }}
    for (int k = 0; k < {net.output_dim}; k++) {{
        float s = {safe}_b2[k];
        for (int j = 0; j < {net.hidden_dim}; j++) {{
            s += hidden[j] * {safe}_w2[j * {net.output_dim} + k];
        }}
        output[k] = s;
    }}
}}
"""
    path = out_dir / f"{name}.h"
    path.write_text(content, encoding="utf-8")
    return path


def export_all_from_npz(
    *,
    weights_dir: Path | None = None,
    out_dir: Path | None = None,
) -> EmbeddedManifest:
    weights_dir = weights_dir or WEIGHTS_DIR
    out_dir = out_dir or EMBEDDED_DIR
    names = ("stability", "maneuver", "mission")
    manifest: dict[str, dict[str, int | list[int]]] = {}
    for name in names:
        net = MLP.load(weights_dir / f"{name}.npz")
        export_mlp_json(net, name, out_dir)
        export_mlp_c_header(net, name, out_dir)
        manifest[name] = {
            "input_dim": net.input_dim,
            "hidden_dim": net.hidden_dim,
            "output_dim": net.output_dim,
        }
    (out_dir / "manifest.json").write_text(
        EmbeddedManifest(manifest).to_json(),
        encoding="utf-8",
    )
    return EmbeddedManifest(manifest)


def forward_embedded_json(model_json: Path, features: np.ndarray) -> np.ndarray:
    """Reference inference matching exported C layout (for validation)."""
    data = json.loads(model_json.read_text(encoding="utf-8"))
    w1 = np.array(data["w1"], dtype=float).reshape(data["w1_shape"])
    w2 = np.array(data["w2"], dtype=float).reshape(data["w2_shape"])
    b1 = np.array(data["b1"], dtype=float)
    b2 = np.array(data["b2"], dtype=float)
    h = np.tanh(features @ w1 + b1)
    return h @ w2 + b2
