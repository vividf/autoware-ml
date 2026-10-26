# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn an autoware-ml ``ptq.ckpt`` into the per-node calibration JSON t4sparse's tools/ptq.py reads.

Only the activation amax is carried over: tools/ptq.py takes weight scales from the
initializer it sees (which is what fold_cpe_linear.py changes), so the checkpoint's weight
amax is not used. Nodes are matched through their filter initializer name
(``<module>.weight``), which both the export and fold_cpe_linear.py keep.

    python ptq_ckpt_to_t4sparse_calib.py ptq.ckpt ptv3_encoder.onnx [ptv3_seg3d_head.onnx ...] -o calib.json
"""

from __future__ import annotations

import argparse
import json

import onnx
import torch

SUFFIX = ".input_quantizer._amax"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    act_amax = {k[: -len(SUFFIX)]: float(v) for k, v in state_dict.items() if k.endswith(SUFFIX)}

    calib: dict[str, dict[str, float]] = {}
    matched: set[str] = set()
    for path in args.onnx:
        graph = onnx.load(path, load_external_data=False).graph
        for node in graph.node:
            if node.op_type != "ImplicitGemm":
                continue
            module = node.input[1].removesuffix(".weight")
            if module not in act_amax:
                print(f"kept fp16 (no calibrated amax): {node.name}")
                continue
            # out_amax is not calibrated here; tools/ptq.py only records it in its report.
            calib[node.name] = {"in_amax": act_amax[module], "out_amax": act_amax[module]}
            matched.add(module)

    # cpe.0 is PTv3's sparse convolution; cpe.1 is the Linear after it (Q/DQ, not a plugin).
    sparse_unmatched = sorted(m for m in act_amax if m.endswith(".cpe.0") and m not in matched)
    if sparse_unmatched:
        raise SystemExit(f"calibrated sparse layers with no ImplicitGemm node: {sparse_unmatched}")
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {len(calib)} node ranges to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
