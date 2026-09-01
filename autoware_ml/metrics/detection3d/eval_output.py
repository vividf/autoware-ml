"""Shared eval-output builder for 3D detection models.

Every detection model decodes its head into per-sample predictions and pairs
them with the ground-truth boxes and labels. This helper builds the flat
eval-output dict that :class:`~autoware_ml.metrics.detection3d.suite.Detection3DMetricSuite`
reads, so each model's ``build_eval_output`` is a one-line delegation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions


def detection_eval_output(
    predictions: list[dict[str, Any]], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair decoded predictions with ground truth for the detection metric.

    Args:
        predictions: Per-sample prediction dicts with ``bboxes_3d``,
            ``scores_3d``, and ``labels_3d``, as returned by ``bbox_head.predict``.
        batch: The batch dictionary holding the ground-truth boxes and labels.

    Returns:
        Flat eval-output dict consumed by the detection metric.
    """
    return {
        "predictions": predictions,
        "gt_boxes": batch["gt_boxes"],
        "gt_labels": batch["gt_labels"],
        "gt_num_points": batch.get("gt_num_points"),
    }


def multi_task_eval_output(
    multi_task_predictions: MultiTaskPredictions, multi_task_batch_inputs: MultiTaskBatchInputs
) -> dict[str, Any]:
    """
    Pair decoded predictions with ground truth for the detection metric.
    This function is a temporary interface between MultiTaskPredictions, MultiTaskFeatures and
    detection_eval_output, and this will be removed once the detection metric is refactored to
    accept MultiTaskPredictions and MultiTaskFeatures directly.

    Args:
        multi_task_predictions: MultiTaskPredictions containing the decoded predictions.
        multi_task_batch_inputs: MultiTaskBatchInputs containing the ground-truth boxes and labels.

    Returns:
        Flat eval-output dict consumed by the detection metric.
    """
    if multi_task_predictions.detection3d_predictions is None:
        raise ValueError(
            "MultiTaskPredictions must contain detection3d_predictions for multi_task_eval_output."
        )

    if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
        raise ValueError(
            "MultiTaskBatchInputs must contain detection3d_gt_batch for multi_task_eval_output."
        )

    gt_detections = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    valid = gt_detections.gt_valid_bboxes
    batch = {
        "gt_boxes": [gt_detections.gt_bboxes_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_labels": [gt_detections.gt_labels_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_num_points": [
            gt_detections.gt_bboxes_num_points[i, : valid[i]] for i in range(len(valid))
        ],
    }

    return detection_eval_output(predictions=multi_task_predictions.to_list(), batch=batch)
