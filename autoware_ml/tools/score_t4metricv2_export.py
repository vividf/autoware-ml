"""Score the T4MetricV2-exported predictions+GT with autoware-ml's MeanAP.

Inputs are the exact DynamicObjects T4MetricV2 evaluated (same predictions,
same GT after its min-points filter), so any difference from V2's own numbers
is purely metric-implementation semantics (matching / AP curve), not data.
"""

from collections import defaultdict

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.structures import Detection3DSample, DetectionState

CLS = ("car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier")
V2_OWN = {
    "car": 0.5318,
    "truck": 0.4281,
    "bus": 0.4967,
    "bicycle": 0.3552,
    "pedestrian": 0.3470,
    "traffic_cone": 0.1817,
    "barrier": 0.1166,
}


def pad9(xy: np.ndarray) -> np.ndarray:
    out = np.zeros((xy.shape[0], 9), dtype=np.float32)
    out[:, :2] = xy
    return out


def main() -> None:
    z = np.load("/workspace/work_dirs/v2_frames_export.npz")
    frames = defaultdict(dict)
    for k in z.files:
        sid, arr = k.rsplit("_", 1)
        frames[sid][arr] = z[k]
    samples = []
    for f in frames.values():
        samples.append(
            Detection3DSample(
                pred_boxes=torch.from_numpy(pad9(f["px"])),
                pred_scores=torch.from_numpy(f["ps"]),
                pred_labels=torch.from_numpy(f["pl"]),
                gt_boxes=torch.from_numpy(pad9(f["gx"])),
                gt_labels=torch.from_numpy(f["gl"]),
            )
        )
    print("frames:", len(samples))
    state = DetectionState(samples=samples, class_names=CLS, thresholds=(0.5, 1.0, 2.0, 4.0))
    rep = MeanAP().evaluate(state, EvalStage.VAL)
    print(f"{'class':<14}{'aml-on-V2data':>14}{'V2 own':>9}{'diff':>9}")
    for c in CLS:
        ap = rep["mAP_" + c]
        print(f"{c:<14}{ap:14.4f}{V2_OWN[c]:9.4f}{ap - V2_OWN[c]:9.4f}")
    print("mAP", round(rep["mAP"], 5), "vs V2 own 0.35103")


if __name__ == "__main__":
    main()
