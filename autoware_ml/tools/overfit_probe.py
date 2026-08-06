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

"""Single-batch overfit probe (autoware-ml side of a cross-framework A/B).

Trains one fixed batch for N steps and writes a per-step loss trace, so the
same probe run in another framework (see ``tools/detection3d/overfit_probe.py``
in AWML) can be compared step by step with
``python -m autoware_ml.tools.compare_overfit``.

To make the trace comparable, every source of run-to-run randomness is off by
default: geometric/photometric augmentation, camera-order shuffle, GridMask,
denoising queries, LR schedule (constant LR), and temporal memory carry-over
(``prev_exists`` is forced to 0 so every step starts from empty memory). Each
can be re-enabled individually to test its contribution.

Reading the result:
  * step-0 losses differ  -> the two frameworks do not see the same input, or
    the forward/loss differs; check the ``meta.fingerprint`` block first.
  * step-0 equal, trajectories diverge -> the optimization path differs
    (parameter groups, gradient scaling, loss normalization).
  * both equal -> the residual gap lives in what this probe disables
    (augmentation distribution, DN, schedule) or in run-to-run noise.

Usage:
    python -m autoware_ml.tools.overfit_probe \\
        --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu \\
        --weights pretrained/awml_t4_best_epoch10_converted.pth \\
        --steps 200 --batch-size 2 \\
        --output parity_out/aml_overfit.jsonl \\
        datamodule.data_root=/workspace/data/t4datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from autoware_ml.configs.paths import CONFIGS_ROOT
from autoware_ml.utils.checkpoints import apply_matching_weights
from autoware_ml.utils import runtime as _runtime  # registers the config resolvers on import

assert _runtime is not None  # keep the resolver-registering import

logger = logging.getLogger(__name__)

# Transforms dropped when augmentation is disabled (matched on _target_ suffix).
_AUGMENTATION_TARGETS = ("GlobalRotScaleTrans", "RandomFlip3D")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse probe arguments, returning the rest as Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Bundled task config name")
    parser.add_argument("--weights", required=True, help="Checkpoint used as the starting point")
    parser.add_argument("--steps", type=int, default=200, help="Optimizer steps on the batch")
    parser.add_argument("--batch-size", type=int, default=2, help="Samples in the fixed batch")
    parser.add_argument(
        "--start-index", type=int, default=0, help="First dataset index of the fixed batch"
    )
    parser.add_argument("--output", required=True, help="JSONL trace output path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Autocast dtype for the forward pass (fp32 keeps the probe deterministic)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override the constant LR")
    parser.add_argument("--keep-augmentation", action="store_true")
    parser.add_argument("--keep-grid-mask", action="store_true")
    parser.add_argument("--keep-dn", action="store_true")
    parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="Carry temporal memory across steps instead of forcing prev_exists=0",
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Allow nondeterministic GPU kernels (faster, but two identical runs "
        "then differ by ~8%% on tail-mean loss, so traces are not comparable)",
    )
    return parser.parse_known_args()


def compose_config(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose the bundled Hydra config the CLI would use."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_ROOT)):
        return compose(config_name=f"tasks/{config_name}", overrides=overrides)


def _pipeline_without_augmentation(pipeline: Any) -> list[Any]:
    """Drop augmentation transforms and pin the resize transform to test mode."""
    kept = []
    for entry in pipeline:
        target = str(entry.get("_target_", ""))
        if any(target.endswith(name) for name in _AUGMENTATION_TARGETS):
            continue
        with open_dict(entry):
            if target.endswith("ResizeCropFlipRotImage"):
                # Test mode: deterministic center crop, no flip.
                entry["training"] = False
                if "data_aug_conf" in entry:
                    entry["data_aug_conf"]["rand_flip"] = False
            if target.endswith("LoadMultiViewImagesFromFiles"):
                entry["shuffle_order"] = False
        kept.append(entry)
    return kept


def apply_determinism(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Strip the configured sources of randomness from the composed config."""
    with open_dict(cfg):
        if not args.keep_augmentation:
            cfg.datamodule.train_transforms.pipeline = _pipeline_without_augmentation(
                cfg.datamodule.train_transforms.pipeline
            )
        if not args.keep_grid_mask:
            cfg.model.use_grid_mask = False
        if not args.keep_dn:
            cfg.model.bbox_head.with_dn = False


def enable_determinism() -> None:
    """Make the run bitwise reproducible so two traces can be compared directly.

    Seeding alone is not enough: it pins step 0 exactly but nondeterministic GPU
    kernels let the 200-step trajectories diverge chaotically. Two identical runs
    differed by 7.9% on the tail-mean loss - the same size as the framework
    differences this probe exists to measure - which makes an unseeded trace
    useless for attributing anything.
    """
    # cuBLAS reads this when it creates its handle, so it has to be set before
    # the first matmul or deterministic matmuls are silently not in effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The flash and memory-efficient SDPA backends have nondeterministic
    # backward passes and use_deterministic_algorithms does not override them -
    # it only warns. They were the whole story here: with them enabled two
    # identical runs diverged at step 1. The math backend is deterministic and
    # mathematically equivalent (it just accumulates differently), which costs
    # memory and speed but is irrelevant at this probe's batch size.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # warn_only: a few detection ops ship no deterministic kernel. Warn and keep
    # going rather than abort - then confirm empirically by running twice and
    # diffing the traces, which is the only check that actually proves it.
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_fixed_batch(
    datamodule: Any, args: argparse.Namespace
) -> tuple[dict[str, Any], list[str]]:
    """Collate a fixed batch straight from the dataset (no sampler, no workers).

    Returns the batch alongside the sample tokens it was built from. The two
    frameworks concatenate scenes in a different order (AWML sorts by
    ``scene_token``, autoware-ml keeps the pkl order), so the same
    ``--start-index`` selects different frames and only the tokens say so.
    """
    datamodule.setup("fit")
    dataset = datamodule.train_dataset
    indices = list(range(args.start_index, args.start_index + args.batch_size))
    if indices[-1] >= len(dataset):
        raise ValueError(
            f"Requested indices {indices} exceed the {len(dataset)}-sample training split."
        )
    infos = getattr(dataset, "data_infos", None)
    tokens = [str(infos[index].get("token")) for index in indices] if infos is not None else []
    return datamodule.train_collate_fn([dataset[index] for index in indices]), tokens


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    """Collect every tensor in an arbitrarily nested batch value.

    Frameworks nest differently (AWML keeps a queue dimension, autoware-ml does
    not), so the fingerprint flattens before summarizing.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if hasattr(value, "tensor") and isinstance(getattr(value, "tensor"), torch.Tensor):
        return [value.tensor]  # mmdet3d box structure
    if isinstance(value, (list, tuple)):
        collected: list[torch.Tensor] = []
        for entry in value:
            collected.extend(flatten_tensors(entry))
        return collected
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [torch.from_numpy(value)]
    except ImportError:  # pragma: no cover
        pass
    return []


# Lidar-frame points projected through every camera. Comparing the resulting
# pixel coordinates is independent of matrix layout and camera ordering, which
# a raw matrix mean is not.
_PROBE_POINTS = ((10.0, 0.0, 0.0), (20.0, 5.0, -1.0), (30.0, -5.0, 0.5))


def projection_probe(value: Any) -> list[list[float]] | None:
    """Project canonical lidar points through every camera matrix."""
    tensors = flatten_tensors(value)
    if not tensors:
        return None
    stacked = torch.cat([tensor.reshape(-1, 4, 4).float() for tensor in tensors], dim=0)
    pixels = []
    for matrix in stacked:
        for point in _PROBE_POINTS:
            projected = matrix[:3] @ torch.tensor([*point, 1.0])
            depth = float(projected[2])
            if depth <= 0.1:
                continue
            pixels.append(
                [round(float(projected[0]) / depth, 2), round(float(projected[1]) / depth, 2)]
            )
    return sorted(pixels)


def fingerprint_batch(batch: dict[str, Any], tokens: list[str] | None = None) -> dict[str, Any]:
    """Summarize the batch so both frameworks can be checked for equal input."""
    report: dict[str, Any] = {}
    if tokens:
        # The one field that identifies the frame outright, so a frame mismatch
        # is a one-line read instead of an inference from gt_counts/timestamps.
        report["tokens"] = list(tokens)
    images = flatten_tensors(batch.get("img"))
    if images:
        image = torch.cat([tensor.reshape(-1, *tensor.shape[-3:]).float() for tensor in images])
        report["img"] = {
            "views": list(image.shape),
            "mean": round(float(image.mean()), 5),
            "std": round(float(image.std()), 5),
            "min": round(float(image.min()), 3),
            "max": round(float(image.max()), 3),
        }
    boxes = batch.get("gt_boxes")
    if boxes is None:
        boxes = batch.get("gt_bboxes_3d")
    box_tensors = [tensor for tensor in flatten_tensors(boxes) if tensor.dim() == 2]
    if box_tensors:
        report["gt_counts"] = [int(tensor.shape[0]) for tensor in box_tensors]
        stacked = torch.cat([tensor.float() for tensor in box_tensors if tensor.numel()], dim=0)
        if stacked.numel():
            report["gt_box_mean"] = [round(float(v), 4) for v in stacked.mean(dim=0)[:7]]
    projection = projection_probe(batch.get("lidar2img"))
    if projection is not None:
        report["projected_pixels"] = projection
    stamps = flatten_tensors(batch.get("timestamp"))
    if stamps:
        values = torch.cat([tensor.reshape(-1).double() for tensor in stamps])
        # AWML stores sequence-relative stamps, autoware-ml absolute epoch
        # seconds, so only the within-batch deltas are comparable.
        report["timestamp_deltas"] = [round(float(v - values[0]), 3) for v in values[:8]]
    return report


def force_stream_reset(batch: dict[str, Any]) -> None:
    """Zero ``prev_exists`` so the head starts each step from empty memory."""
    value = batch.get("prev_exists")
    if isinstance(value, torch.Tensor):
        batch["prev_exists"] = torch.zeros_like(value)
    elif isinstance(value, (list, tuple)):
        batch["prev_exists"] = [torch.zeros_like(entry) for entry in value]


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch container onto ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_to_device(item, device) for item in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value


def build_optimizer(
    model: Any, cfg: DictConfig, lr_override: float | None
) -> torch.optim.Optimizer:
    """Build a constant-LR AdamW with the configured per-group learning rates."""
    optimizer_cfg = cfg.model.optimizer
    base_lr = lr_override if lr_override is not None else float(optimizer_cfg.lr)
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    overrides = cfg.model.get("optimizer_group_overrides", {}) or {}
    groups = model.build_optimizer_groups()
    param_groups = []
    for name, params in groups.items():
        if not params:
            continue
        group_lr = base_lr
        override = overrides.get(name)
        if override is not None and override.get("lr") is not None:
            scale = float(override.lr) / float(optimizer_cfg.lr)
            group_lr = base_lr * scale
        param_groups.append({"params": params, "lr": group_lr, "name": name})
    logger.info(
        "Optimizer groups: %s",
        {group["name"]: {"n": len(group["params"]), "lr": group["lr"]} for group in param_groups},
    )
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run_probe() -> None:
    """Overfit one fixed batch and write the per-step loss trace."""
    args, overrides = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    L.seed_everything(args.seed, workers=True)
    if args.nondeterministic:
        torch.backends.cudnn.benchmark = False
    else:
        enable_determinism()

    cfg = compose_config(args.config_name, overrides)
    apply_determinism(cfg, args)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    batch, tokens = build_fixed_batch(datamodule, args)
    fingerprint = fingerprint_batch(batch, tokens)
    logger.info("Batch fingerprint: %s", json.dumps(fingerprint, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model)
    apply_matching_weights(model, args.weights, map_location="cpu", logger=logger)
    model.to(device).train()

    batch = move_to_device(batch, device)
    if not args.keep_memory:
        force_stream_reset(batch)

    optimizer = build_optimizer(model, cfg, args.lr)
    clip_value = cfg.trainer.get("gradient_clip_val")
    autocast_dtype = _AUTOCAST_DTYPES.get(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        header = {
            "framework": "autoware-ml",
            "config": args.config_name,
            "weights": args.weights,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "start_index": args.start_index,
            "precision": args.precision,
            "augmentation": args.keep_augmentation,
            "grid_mask": args.keep_grid_mask,
            "dn": args.keep_dn,
            "memory_carry": args.keep_memory,
            "deterministic": not args.nondeterministic,
            "fingerprint": fingerprint,
        }
        stream.write(json.dumps({"meta": header}) + "\n")

        forward_keys = [key for key in model.forward_signature.parameters if key in batch]
        for step in range(args.steps):
            if not args.keep_memory:
                model.bbox_head.reset_memory()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(**{key: batch[key] for key in forward_keys})
                metrics = model.compute_metrics(batch, outputs)
            loss = metrics["loss"]
            scaler.scale(loss).backward()
            if clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_value))
            scaler.step(optimizer)
            scaler.update()

            record = {
                "step": step,
                **{
                    key: round(float(value.detach()), 6)
                    if isinstance(value, torch.Tensor)
                    else round(float(value), 6)
                    for key, value in metrics.items()
                },
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if step % 10 == 0 or step == args.steps - 1:
                logger.info("step %4d  loss=%.5f", step, float(loss))

    logger.info("Wrote %d steps to %s", args.steps, output_path)


if __name__ == "__main__":
    run_probe()
