#!/usr/bin/env python3
"""A stand-in perceptive policy for testing r1_ctrl's Perceptive state before one is trained:
it takes the perceptive input (the flat policy's 83 observations plus the 187-value height scan)
and runs the flat policy on the first 83, ignoring the map. It walks like the flat policy and
cannot climb; everything around it (height maps, fallback, head tilt, walking service) runs for real.

    .venv/bin/python tools/make_test_policy.py
writes deploy/robots/r1/config/policy/perceptive/v0/{exported/policy.onnx, params/deploy.yaml}.
A trained policy goes into a later folder (v1): r1_ctrl takes the last one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2] / "deploy/robots/r1/config/policy"
FLAT = ROOT / "velocity/v0"
OUT = ROOT / "perceptive/v0"
SCAN = 17 * 11


def main() -> None:
    flat = onnx.load(FLAT / "exported/policy.onnx")
    (obs,) = flat.graph.input
    n_flat = obs.type.tensor_type.shape.dim[1].dim_value
    graph = flat.graph
    # New input "obs" of the perceptive width; the flat graph reads its first n_flat columns.
    for node in graph.node:
        node.input[:] = ["obs_flat" if i == obs.name else i for i in node.input]
    graph.input.remove(obs)
    graph.input.append(helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_flat + SCAN]))
    graph.initializer.extend([
        helper.make_tensor("slice_start", TensorProto.INT64, [1], [0]),
        helper.make_tensor("slice_end", TensorProto.INT64, [1], [n_flat]),
        helper.make_tensor("slice_axis", TensorProto.INT64, [1], [1]),
    ])
    graph.node.insert(0, helper.make_node("Slice", ["obs", "slice_start", "slice_end", "slice_axis"], ["obs_flat"]))
    flat.metadata_props.add(key="test_policy", value="flat policy on the first %d inputs; height scan ignored" % n_flat)
    onnx.checker.check_model(flat)
    (OUT / "exported").mkdir(parents=True, exist_ok=True)
    onnx.save(flat, OUT / "exported/policy.onnx")

    # deploy.yaml: the flat one plus the height scan, last, as in Unitree-R1-Perceptive's actor.
    (OUT / "params").mkdir(parents=True, exist_ok=True)
    text = (FLAT / "params/deploy.yaml").read_text()
    if not text.endswith("\n"):
        text += "\n"
    scale = "".join("    - 0.2\n" for _ in range(SCAN))
    text += (
        "  height_scan:\n"
        "    params:\n"
        "      width: 17\n      height: 11\n      resolution: 0.1\n      origin: [-0.8, -0.5]\n"
        "      max_age_s: 0.3\n      fallback_height: 0.74\n      topic: rt/heightmap\n"
        "    clip: null\n"
        "    scale:\n" + scale +
        "    history_length: 1\n"
    )
    (OUT / "params/deploy.yaml").write_text(text)
    print(f"wrote {OUT}: input {n_flat} + {SCAN}, output as the flat policy")


if __name__ == "__main__":
    main()
