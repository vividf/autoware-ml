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

"""Compare two overfit-probe traces and say where the frameworks diverge.

Reads the JSONL traces written by ``autoware_ml.tools.overfit_probe`` (this
repo) and ``tools/detection3d/overfit_probe.py`` (AWML), normalizes the loss
key names, and reports:

  1. whether the two probes saw the same input (batch fingerprint),
  2. the step-0 losses (forward + loss agreement at identical weights),
  3. the loss trajectories at checkpoints (optimization agreement),
  4. the per-loss-term breakdown at the last common step.

Uses only the standard library, so it runs on the host (no container needed).

Usage:
    python -m autoware_ml.tools.compare_overfit \\
        parity_out/aml_overfit.jsonl \\
        ../AWML/parity_out/awml_overfit.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# AWML prefixes its per-frame loss keys; autoware-ml does not.
_KEY_PREFIXES = ("frame_0_",)
# Loss terms that mean the same thing under different names.
_KEY_ALIASES = {
    "enc_loss_cls": "loss_cls2d",
    "enc_loss_bbox": "loss_bbox2d",
    "enc_loss_iou": "loss_iou2d",
    "centers2d_losses": "loss_centers2d",
    "centerness_losses": "loss_centerness2d",
}


def normalize_key(key: str) -> str:
    """Strip framework-specific prefixes and map aliases onto one name."""
    for prefix in _KEY_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return _KEY_ALIASES.get(key, key)


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Load one probe trace into its metadata header and per-step records."""
    meta: dict[str, Any] = {}
    steps: list[dict[str, float]] = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if "meta" in record:
                meta = record["meta"]
                continue
            steps.append({normalize_key(key): value for key, value in record.items()})
    if not steps:
        raise ValueError(f"{path} contains no step records.")
    return meta, steps


def format_delta(left: float | None, right: float | None) -> str:
    """Render a signed difference, or a placeholder when a side is missing."""
    if left is None or right is None:
        return "     -"
    return f"{left - right:+.4f}"


# Per-key numeric tolerance for the fingerprint comparison. The image
# statistics leave room for residual decode differences only - a resize kernel
# mismatch (PIL's antialiased bicubic vs cv2's INTER_LINEAR) moves the pixel
# std by ~4% at this pipeline's downscale and is meant to be caught, not
# absorbed. The projected pixels absorb sub-pixel intrinsics differences.
_FINGERPRINT_TOLERANCE = {
    "img": 5e-3,
    "gt_box_mean": 2e-3,
    "projected_pixels": 0.5,
    "timestamp_deltas": 1e-3,
    "gt_counts": 0.0,
}

# Layout of the 7-element box mean: x, y, z, dx, dy, dz, yaw.
_BOX_MEAN_Z = 2
_BOX_MEAN_HEIGHT = 5


def compare_box_mean(left: Any, right: Any, tolerance: float) -> tuple[bool, str]:
    """Compare box means, absorbing a z-origin convention difference.

    autoware-ml reports the gravity-centre z where AWML reports the bottom-face
    z, so the two differ by exactly half the mean box height. That is a
    representation difference, not a disagreement about the boxes. Every other
    component is origin-invariant and still has to match outright.
    """
    if not (isinstance(left, list) and isinstance(right, list)):
        return left == right, "structural comparison"
    if len(left) != len(right) or len(left) <= _BOX_MEAN_HEIGHT:
        return left == right, "structural comparison"
    invariant = max(
        abs(a - b) for index, (a, b) in enumerate(zip(left, right)) if index != _BOX_MEAN_Z
    )
    half_height = (left[_BOX_MEAN_HEIGHT] + right[_BOX_MEAN_HEIGHT]) / 4
    z_delta = left[_BOX_MEAN_Z] - right[_BOX_MEAN_Z]
    candidates = {
        "same z origin": abs(z_delta),
        "gravity-centre vs bottom-face z": min(
            abs(z_delta - half_height), abs(z_delta + half_height)
        ),
    }
    label, z_residual = min(candidates.items(), key=lambda item: item[1])
    worst = max(invariant, z_residual)
    return worst <= tolerance, f"max|delta|={worst:.4g} (tol {tolerance:g}, {label})"


def _numbers(value: Any) -> list[float]:
    """Flatten a fingerprint entry into comparable numbers."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        # Channel order differs by design (RGB vs BGR), so the per-channel
        # layout is not compared - only the order-invariant statistics.
        return [float(value[key]) for key in ("mean", "std") if key in value]
    if isinstance(value, (list, tuple)):
        collected: list[float] = []
        for entry in value:
            collected.extend(_numbers(entry))
        return collected
    return []


def compare_fingerprints(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Print the batch fingerprints and report whether the inputs agree."""
    print("=" * 78)
    print("1. INPUT CHECK - did both probes see the same batch?")
    print("=" * 78)
    identical = True
    differing: list[str] = []
    for key in sorted(set(left) | set(right)):
        left_value, right_value = left.get(key), right.get(key)
        left_numbers, right_numbers = _numbers(left_value), _numbers(right_value)
        tolerance = _FINGERPRINT_TOLERANCE.get(key, 1e-6)
        if key == "gt_box_mean":
            match, detail = compare_box_mean(left_value, right_value, tolerance)
        elif left_numbers and len(left_numbers) == len(right_numbers):
            worst = max(abs(a - b) for a, b in zip(left_numbers, right_numbers))
            match = worst <= tolerance
            detail = f"max|delta|={worst:.4g} (tol {tolerance:g})"
        else:
            match = left_value == right_value
            detail = "structural comparison"
        identical = identical and bool(match)
        print(f"  [{'OK  ' if match else 'DIFF'}] {key:<18} {detail}")
        if not match:
            differing.append(key)
            print(f"         aml : {left_value}")
            print(f"         awml: {right_value}")
    if identical:
        print("\n  -> Inputs agree. Loss differences below are model/optimization side.")
        return True
    print("\n  -> Inputs DIFFER. Fix this first; loss comparisons are not meaningful yet.")
    # Only the hints for keys that actually differ, so the diagnosis is not
    # buried in advice about things that matched.
    if "tokens" in differing:
        print("     tokens differ -> different frames. The two frameworks concatenate")
        print("       scenes in a different order (AWML sorts by scene_token,")
        print("       autoware-ml keeps pkl order), so an equal --start-index is NOT")
        print("       the same frame. Map one index to the other via the token.")
    if {"timestamp_deltas", "gt_counts"} & set(differing):
        print("     timestamp_deltas / gt_counts differ -> different frames were selected.")
    if "projected_pixels" in differing:
        print("     projected_pixels differ -> camera geometry (crop/resize/intrinsics) differs.")
    if differing == ["img"]:
        print("     only img statistics differ -> same frame and same geometry, so this is")
        print("       image preprocessing: a resize interpolation kernel or decode mismatch.")
    return identical


def compare_trajectories(
    left: list[dict[str, float]], right: list[dict[str, float]], term: str
) -> None:
    """Print the loss trajectory of both traces at log-spaced checkpoints."""
    print()
    print("=" * 78)
    print(f"2. TRAJECTORY - '{term}' over steps")
    print("=" * 78)
    common = min(len(left), len(right))
    marks = sorted({0, 1, 2, 4, 9, 19, 49, 99, 199, common - 1} & set(range(common)))
    print(f"  {'step':>6}  {'aml':>10}  {'awml':>10}  {'aml-awml':>10}  {'rel':>7}")
    for index in marks:
        left_value = left[index].get(term)
        right_value = right[index].get(term)
        if left_value is None or right_value is None:
            continue
        relative = (left_value - right_value) / max(abs(right_value), 1e-9)
        print(
            f"  {index:>6}  {left_value:>10.4f}  {right_value:>10.4f}  "
            f"{left_value - right_value:>+10.4f}  {relative:>+6.1%}"
        )
    first_left, first_right = left[0].get(term), right[0].get(term)
    last_left, last_right = left[common - 1].get(term), right[common - 1].get(term)
    if None not in (first_left, first_right, last_left, last_right):
        print()
        print(f"  step-0 gap      : {format_delta(first_left, first_right)}")
        print(f"  final gap       : {format_delta(last_left, last_right)}")
        print(f"  aml   reduction : {first_left - last_left:+.4f}")
        print(f"  awml  reduction : {first_right - last_right:+.4f}")
        print()
        if abs(first_left - first_right) > 0.05 * max(abs(first_right), 1e-9):
            print("  VERDICT: step-0 already differs -> forward/loss or input difference.")
        elif abs(last_left - last_right) > 0.05 * max(abs(last_right), 1e-9):
            print("  VERDICT: step-0 matches but trajectories diverge -> optimization side")
            print("           (parameter groups, LR resolution, gradient handling, loss norm).")
        else:
            print("  VERDICT: both match -> the residual gap is NOT in what this probe covers;")
            print("           re-enable a disabled component (--keep-augmentation / --keep-dn /")
            print("           --keep-grid-mask / --keep-memory) and re-run to bisect it.")


def compare_terms(left: list[dict[str, float]], right: list[dict[str, float]]) -> None:
    """Print the per-term breakdown at step 0 and at the last common step."""
    common = min(len(left), len(right))
    print()
    print("=" * 78)
    print("3. PER-TERM BREAKDOWN")
    print("=" * 78)
    terms = sorted((set(left[0]) | set(right[0])) - {"step"})
    print(
        f"  {'term':<24}{'aml@0':>9}{'awml@0':>9}{'d@0':>9}   {'aml@N':>9}{'awml@N':>9}{'d@N':>9}"
    )
    for term in terms:
        l0, r0 = left[0].get(term), right[0].get(term)
        ln, rn = left[common - 1].get(term), right[common - 1].get(term)
        if l0 is None and r0 is None:
            continue
        cells = [
            f"{l0:>9.4f}" if l0 is not None else "        -",
            f"{r0:>9.4f}" if r0 is not None else "        -",
            f"{format_delta(l0, r0):>9}",
            f"{ln:>9.4f}" if ln is not None else "        -",
            f"{rn:>9.4f}" if rn is not None else "        -",
            f"{format_delta(ln, rn):>9}",
        ]
        print(
            f"  {term:<24}"
            + cells[0]
            + cells[1]
            + cells[2]
            + "   "
            + cells[3]
            + cells[4]
            + cells[5]
        )
    print()
    print("  Terms present on only one side are naming differences, not missing losses;")
    print("  add them to _KEY_ALIASES if a pair should line up.")


def main() -> None:
    """Compare two probe traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aml_trace", type=Path, help="autoware-ml probe JSONL")
    parser.add_argument("awml_trace", type=Path, help="AWML probe JSONL")
    parser.add_argument("--term", default="loss", help="Loss term for the trajectory table")
    args = parser.parse_args()

    aml_meta, aml_steps = load_trace(args.aml_trace)
    awml_meta, awml_steps = load_trace(args.awml_trace)

    print("probe settings")
    for key in (
        "framework",
        "weights",
        "steps",
        "batch_size",
        "start_index",
        "precision",
        "augmentation",
        "grid_mask",
        "dn",
        "memory_carry",
        "deterministic",
    ):
        print(f"  {key:<14} aml={aml_meta.get(key)!s:<28} awml={awml_meta.get(key)!s}")
    print()
    for key in ("precision", "batch_size", "start_index", "dn", "grid_mask", "augmentation"):
        if aml_meta.get(key) != awml_meta.get(key):
            print(
                f"  WARNING: '{key}' differs between the two probes - not an apples-to-apples run."
            )
    # A nondeterministic trace cannot support attribution: two identical runs
    # drift ~8% on the tail-mean loss, which swamps the differences being
    # measured. Only step 0 stays trustworthy in that case.
    if not all(meta.get("deterministic") for meta in (aml_meta, awml_meta)):
        print("  WARNING: at least one trace is NOT deterministic (or predates the flag).")
        print("           Trust step 0 only; treat trajectory and per-term gaps below as")
        print("           unattributable noise until both sides are re-run deterministically.")
    print()

    compare_fingerprints(aml_meta.get("fingerprint", {}), awml_meta.get("fingerprint", {}))
    compare_trajectories(aml_steps, awml_steps, args.term)
    compare_terms(aml_steps, awml_steps)


if __name__ == "__main__":
    main()
