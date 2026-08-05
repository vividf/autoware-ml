# Metrics

> **What this covers:** the concrete metrics — the detection and segmentation suites, their
> components (mAP, NDS, IoU…), range-awareness, key naming, config, and how to add one.
> Prerequisite: [evaluation_pipeline.md](evaluation_pipeline.md) (the suite/metric machinery).

---

## 1. Built-in suites and their metrics

| Suite | `prefix` | `_required_keys` | Components |
| ----- | -------- | ---------------- | ---------- |
| `Detection3DMetricSuite` (`metrics/detection3d/suite.py`) | `det3d` | `predictions`, `gt_boxes`, `gt_labels` | `MeanAP`, `HeadingAP`, `Nds`, `TpErrors` |
| `Segmentation3DMetricSuite` (`metrics/segmentation3d/suite.py`) | `seg3d` | `seg_pred_labels`, `seg_target_labels`, `seg_coord` | `IoU`, `Accuracy`, `PrecisionRecallF1` |

A model's `model.metrics` is a **list** of suites — a joint seg+det model lists two. Each
suite gets its `components` (which metrics to run) and each component declares its `stages`.

---

## 2. Range-awareness (the distinctive feature)

Both suites are **range-aware**: you configure radial `MetricRange` windows, and every metric
key is *also* emitted per range with a distance suffix. So one config yields:

```text
test/det3d/mAP                 (overall)
test/det3d/mAP_car             (per class, overall)
test/det3d/mAP_car_0m_50m      (per class, per range)
test/det3d/mAP_car_50m_90m
...
```

`MetricRange` (`metrics/base.py:37`) is `{name, min_distance, max_distance}` (max `None` =
unbounded). Detection clips boxes per range; segmentation keeps one confusion matrix per range
and buckets points by `seg_coord`. Range suffixes must be unique or the suite raises at
construction.

Why this matters for AV perception: a detector that's great at 30 m but poor at 90 m is a
safety problem the overall mAP hides. Per-range metrics make that visible.

---

## 3. `MeanAP` up close (`metrics/detection3d/mean_ap.py`)

```python
class MeanAP(Metric[DetectionState]):
    def evaluate(self, state, stage):
        full = stage is EvalStage.TEST
        labels = state.labels(full)
        if not labels:
            return {} if stage is EvalStage.VAL else {"mAP": float("nan")}

        # per-class AP = mean over center-distance thresholds
        per_class_ap = {
            label: mean_valid([curve_metrics(state.match_curve(label, t)).ap for t in state.thresholds])
            for label in labels
        }
        report = {"mAP": mean_valid(list(per_class_ap.values()))}
        for label, ap in per_class_ap.items():
            report[f"mAP_{label_metric_name(label, state.class_names)}"] = ap
        if stage is EvalStage.VAL:
            return report                              # validation = cheap: only mAP + per-class AP

        # test adds per-class GT count and the full per-threshold curve details
        for label in labels:
            name = label_metric_name(label, state.class_names)
            report[f"gt_count_{name}"] = float(state.match_curve(label, state.thresholds[0]).total_gt)
            for threshold in state.thresholds:
                curve = state.match_curve(label, threshold); m = curve_metrics(curve)
                token = threshold_token(threshold)
                report[f"AP_{name}_{token}"] = m.ap
                report[f"num_match_{name}_{token}"] = float(curve.num_match)
                report[f"max_f1_{name}_{token}"] = m.max_f1
                report[f"optimal_conf_{name}_{token}"] = m.optimal_conf
                # ... optimal recall/precision ...
        return report
```

This is nuScenes-style AP: matching is by **center distance** at thresholds
`[0.5, 1.0, 2.0, 4.0]` m (not IoU), AP is averaged over thresholds, and mAP is the class mean.
The `stage` split (val = headline only, test = full curves) is the "cheap val / full test"
convention in action.

The other detection components:

| Metric | File | Stages (typical) | What it adds |
| ------ | ---- | ---------------- | ------------ |
| `MeanAP` | `mean_ap.py` | `val`, `test` | mAP + per-class AP (+ curves at test) |
| `Nds` | `nds.py` | `test` | nuScenes Detection Score (combines mAP/APH + TP errors) |
| `HeadingAP` | `heading_ap.py` | `test` | orientation-aware AP |
| `TpErrors` | `tp_errors.py` | `test` | translation/scale/orientation/velocity errors of true positives |

Segmentation (`Segmentation3DMetricSuite`): `IoU`, `Accuracy`, `PrecisionRecallF1`, all derived
from a single accumulated `(ranges+1, C, C)` confusion matrix (reduced with `sum` across GPUs).

---

## 4. Configuring the suite

Metrics are defined in the **dataset group** config (so class names/ranges come from one
place) and referenced by the model via `metrics: ${dataset.detection3d.metrics}`. A suite is a
`_target_` with a `components` list:

```yaml
model:
  metrics:
    - _target_: autoware_ml.metrics.detection3d.suite.Detection3DMetricSuite
      class_names: ${class_names}
      eval_class_range: ${metric_eval_class_range}   # per-class distance caps
      ranges: ${metric_ranges}
      components:
        - { _target_: autoware_ml.metrics.detection3d.mean_ap.MeanAP,     stages: [val, test] }
        - { _target_: autoware_ml.metrics.detection3d.heading_ap.HeadingAP, stages: [test] }
        - { _target_: autoware_ml.metrics.detection3d.nds.Nds,            stages: [test] }
        - { _target_: autoware_ml.metrics.detection3d.tp_errors.TpErrors, stages: [test] }
```

### The retune-without-restating trick

`model.metrics` is a **list**, and Hydra replaces a list wholesale (it doesn't merge). So to
retune a variant you would otherwise have to restate the whole suite. The framework avoids this
by having the suite read its tunable bits from two interpolation variables:

```yaml
# base config
metric_ranges:
  - { _target_: autoware_ml.metrics.base.MetricRange, name: 0-50m, min_distance: 0.0, max_distance: 50.0 }
  - { _target_: autoware_ml.metrics.base.MetricRange, name: 50-90m, min_distance: 50.0, max_distance: 90.0 }
metric_eval_class_range: { car: 121.0, truck: 121.0, pedestrian: 121.0, ... }

# a variant retunes by overriding ONLY these two variables — the suite definition is untouched
metric_eval_class_range: { car: 102.0, pedestrian: 102.0, ... }
```

---

## 5. Writing a custom metric

A metric is the unit of extension — subclass `Metric`, declare `stages`, read the suite's
`state`:

```python
from autoware_ml.metrics.base import Metric, EvalStage

class PerClassRecall(Metric):
    def evaluate(self, state, stage: EvalStage) -> dict[str, float]:
        return {
            f"recall_class_{i}": float(state.recall[i].item())
            for i in range(state.num_classes)
            if bool(state.has_support[i])
        }
```

Then add it to the suite's `components` list in config — its keys appear under the suite's
prefix (and per range automatically). No suite edit needed. If your metric needs *new state*
the suite doesn't build, that's a new `MetricSuite` instead.

---

## 6. Reading the numbers

- Keys land in MLflow as `{split}/{prefix}/{key}`. Compare `val/det3d/mAP` across runs; drill
  into `test/det3d/mAP_pedestrian_50m_90m` to find where a model is weak.
- `autoware-ml test --config-name <cfg> --weights <best.ckpt>` produces the full test report
  (single device by default → exact).

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `... was constructed with no components` warning | suite has an empty `components` list | add metric components |
| `Range metric suffixes must be unique` | two `MetricRange`s produce the same suffix | make ranges distinct |
| Retuning ranges restates the whole suite | overriding `model.metrics` (a list) directly | override `metric_ranges`/`metric_eval_class_range` variables instead |
| mAP lower than expected vs another repo | matching is center-distance @ `[0.5,1,2,4]` m, per-class range caps | confirm thresholds and `eval_class_range` match the baseline |
| Per-range keys missing | `ranges` not configured / empty | set `ranges: ${metric_ranges}` |
| Metric key collides | two metrics emit the same name | give one a distinct name/prefix |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Change distance buckets | edit `metric_ranges` |
| Change per-class eval range | edit `metric_eval_class_range` |
| Run NDS/TP errors at val too | add `val` to their `stages` (costs val-epoch time) |
| Add a metric | write a `Metric`, add to `components` |
| Add a new task's metrics | write a `MetricSuite` (+ its `Metric`s) with a new `prefix` and `_required_keys` |
| Monitor a metric for checkpoints/Optuna | point `monitor`/`optimized_metric` at `{split}/{prefix}/{key}` |

---

**Next (Phase 6):** [../deployment/export_pipeline.md](../deployment/export_pipeline.md) — turning
a trained checkpoint into an ONNX / TensorRT artifact for Autoware.
