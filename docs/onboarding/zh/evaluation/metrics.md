# 指標 (Metrics)

> **本文涵蓋內容：** 具體的指標 — 偵測（detection）與分割（segmentation）的 suite、其組成元件
> （mAP、NDS、IoU 等）、範圍感知（range-awareness）、key 命名方式、config，以及如何新增一個指標。
> 先備知識：[evaluation_pipeline.md](evaluation_pipeline.md)（suite/metric 的運作機制）。

---

## 1. 內建的 suite 及其指標

| Suite | `prefix` | `_required_keys` | 組成元件（Components） |
| ----- | -------- | ---------------- | ---------- |
| `Detection3DMetricSuite` (`metrics/detection3d/suite.py`) | `det3d` | `predictions`, `gt_boxes`, `gt_labels` | `MeanAP`, `HeadingAP`, `Nds`, `TpErrors` |
| `Segmentation3DMetricSuite` (`metrics/segmentation3d/suite.py`) | `seg3d` | `seg_pred_labels`, `seg_target_labels`, `seg_coord` | `IoU`, `Accuracy`, `PrecisionRecallF1` |

模型的 `model.metrics` 是一個由多個 suite 組成的**list** — 一個同時具備 seg+det 的聯合模型會
列出兩個 suite。每個 suite 都有各自的 `components`（決定要執行哪些指標），而每個 component
也會宣告自己的 `stages`。

---

## 2. 範圍感知（Range-awareness）（此設計的特色）

兩個 suite 都具備**範圍感知（range-aware）**能力：你可以設定放射狀（radial）的 `MetricRange`
區間，而每個指標 key *也* 會依範圍以距離後綴（suffix）的形式輸出。因此，單一個 config 就能
產生：

```text
test/det3d/mAP                 (overall)
test/det3d/mAP_car             (per class, overall)
test/det3d/mAP_car_0m_50m      (per class, per range)
test/det3d/mAP_car_50m_90m
...
```

`MetricRange`（`metrics/base.py:37`）是 `{name, min_distance, max_distance}`（`max` 為 `None`
代表無上限）。Detection 會依範圍裁切（clip）boxes；segmentation 則為每個範圍保留一個混淆矩陣，
並依 `seg_coord` 將點分類（bucket）。範圍後綴必須是唯一的，否則 suite 會在建構時拋出錯誤。

這對自動駕駛（AV）感知為何重要：一個在 30 公尺表現優異、卻在 90 公尺表現不佳的偵測器，是整體
mAP 會掩蓋掉的安全性問題。依範圍區分的指標能讓這個問題顯現出來。

---

## 3. 近距離觀察 `MeanAP` (`metrics/detection3d/mean_ap.py`)

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

這是 nuScenes 風格的 AP：配對（matching）的依據是**中心距離（center distance）**，門檻值為
`[0.5, 1.0, 2.0, 4.0]` 公尺（而非 IoU），AP 是各門檻值的平均，而 mAP 則是各類別的平均。`stage`
的區分方式（val = 僅核心指標，test = 完整曲線）正是「val 從簡、test 求全」這項慣例的實際體現。

其他 detection 的組成元件：

| Metric | 檔案 | Stages（一般情況） | 新增內容 |
| ------ | ---- | ---------------- | ------------ |
| `MeanAP` | `mean_ap.py` | `val`、`test` | mAP + 各類別 AP（test 時另加曲線） |
| `Nds` | `nds.py` | `test` | nuScenes Detection Score（結合 mAP/APH + TP 誤差） |
| `HeadingAP` | `heading_ap.py` | `test` | 具方向感知（orientation-aware）的 AP |
| `TpErrors` | `tp_errors.py` | `test` | true positive 的平移（translation）/尺度（scale）/方向（orientation）/速度（velocity）誤差 |

Segmentation（`Segmentation3DMetricSuite`）：`IoU`、`Accuracy`、`PrecisionRecallF1` 皆源自單一個
累積的 `(ranges+1, C, C)` 混淆矩陣（跨 GPU 以 `sum` 歸約）。

---

## 4. 設定 suite

指標是定義在**dataset group** 的 config 中（如此一來，類別名稱/範圍就統一來自單一位置），並由
模型透過 `metrics: ${dataset.detection3d.metrics}` 來引用。一個 suite 就是一個帶有
`components` list 的 `_target_`：

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

### 「重新調整而不需重述」的技巧

`model.metrics` 是一個**list**，而 Hydra 對 list 的處理方式是整個替換（不會合併）。因此若要
調整某個變體（variant），原本會需要重述整個 suite。框架透過讓 suite 從兩個插值（interpolation）
變數中讀取可調整的部分，來避免這個問題：

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

## 5. 撰寫自訂指標

指標是擴充的最小單位 — 繼承 `Metric`、宣告 `stages`、讀取 suite 的 `state`：

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

接著將其加入 config 中 suite 的 `components` list — 它的 key 就會出現在該 suite 的 prefix
之下（並自動依範圍區分）。完全不需要修改 suite。如果你的指標需要 suite 尚未建構的*新 state*，
那就應該改為建立一個新的 `MetricSuite`。

---

## 6. 解讀數值

- Key 會以 `{split}/{prefix}/{key}` 的形式出現在 MLflow 中。可以跨多次執行（run）比較
  `val/det3d/mAP`；深入查看 `test/det3d/mAP_pedestrian_50m_90m` 以找出模型表現較弱之處。
- `autoware-ml test --config-name <cfg> --weights <best.ckpt>` 會產生完整的 test 報告
  （預設為單一裝置 → 精確數值）。

---

## 常見除錯情境

| 症狀 | 原因 | 修正方式 |
| ------- | ----- | --- |
| `... was constructed with no components` 警告 | suite 的 `components` list 是空的 | 新增指標的 components |
| `Range metric suffixes must be unique` | 兩個 `MetricRange` 產生了相同的後綴 | 讓範圍彼此不同 |
| 重新調整範圍時，需要重述整個 suite | 直接覆寫 `model.metrics`（一個 list） | 改為覆寫 `metric_ranges`/`metric_eval_class_range` 變數 |
| mAP 比其他 repo 預期的低 | 配對方式是以 `[0.5,1,2,4]` 公尺的中心距離為準，且各類別有範圍上限 | 確認門檻值與 `eval_class_range` 是否與 baseline 一致 |
| 依範圍區分的 key 消失 | `ranges` 未設定 / 為空 | 設定 `ranges: ${metric_ranges}` |
| 指標的 key 發生衝突 | 兩個指標輸出了相同的名稱 | 為其中一個指定不同的名稱/prefix |

---

## 常見修改情境

| 我想要… | 這麼做 |
| ---------- | ------- |
| 變更距離分桶（bucket） | 編輯 `metric_ranges` |
| 變更各類別的評估範圍 | 編輯 `metric_eval_class_range` |
| 讓 NDS/TP 誤差也在 val 時執行 | 將 `val` 加入它們的 `stages`（會增加 val epoch 的耗時） |
| 新增一個指標 | 撰寫一個 `Metric`，加入 `components` |
| 為新任務新增指標 | 撰寫一個 `MetricSuite`（以及其對應的 `Metric`），並指定新的 `prefix` 與 `_required_keys` |
| 為 checkpoints/Optuna 監控某個指標 | 將 `monitor`/`optimized_metric` 指向 `{split}/{prefix}/{key}` |

---

**下一步（第 6 階段）：** [../deployment/export_pipeline.md](../deployment/export_pipeline.md) — 將
訓練好的 checkpoint 轉換為供 Autoware 使用的 ONNX / TensorRT 產出物（artifact）。
