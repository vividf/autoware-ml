# Autoware-ML 新手上路指南

> 這是一份以學習為導向的 Autoware-ML 框架導覽，專為新貢獻者撰寫。
> 目標：在 1 到 2 週內，你應該能夠獨立**讀懂大部分程式碼、新增一個模型，
> 並且排除訓練或部署上的問題**。

本指南**並非**要取代 `docs/framework/`、`docs/user-guide/` 與 `docs/models/` 底下的參考文件。
那些文件告訴你 *API 是什麼*；而本指南要告訴你的是 *框架為什麼是這個樣子* 以及 *各個部分如何組合在一起*，
這樣一來，參考文件才會開始變得有意義。

這裡的所有內容都以原始碼為根據。當某個說法指向程式碼時，會使用可以直接開啟的
`path:line` 參照。**原始碼才是真相來源** — 如果本指南與程式碼有出入，一律以程式碼為準，
請協助修正本指南。

---

## 這份指南適合誰

適合以下這樣的工程師：

- 熟悉 **PyTorch**（tensor、`nn.Module`、autograd、訓練迴圈），
- 具備一些**自動駕駛／3D 感知**背景（點雲、bounding box、相機），
- 但**對這個框架還是新手** — 而且可能也是第一次接觸 PyTorch Lightning 和 Hydra。

如果你是從舊版的 `tier4/AWML`（以 MMDetection3D 為基礎）repo 過來的，
請先閱讀 [architecture/framework_overview.md](architecture/framework_overview.md) —
這裡的思維模型完全不同，沿用你在 AWML 的直覺反而會讓你誤入歧途。

---

## 五個層次的心智模型

你不需要一次搞懂所有東西，可以分層學習。每一個層次都是下一個層次的前提。

| 層次 | 你會理解… | 閱讀 |
| ----- | --------------- | ---- |
| **1. 框架結構** | 什麼東西放在哪裡、為什麼這樣放；技術堆疊（Lightning + Hydra + MLflow） | [architecture/framework_overview.md](architecture/framework_overview.md) |
| **2. 資料流** | 一筆 sample 如何變成一個可送進 GPU 的 batch | [architecture/data_flow.md](architecture/data_flow.md)，接著看 [dataset/](dataset/dataset_pipeline.md) |
| **3. 訓練流程** | `trainer.fit()` 如何把一個 batch 轉換成一次權重更新 | [architecture/execution_flow.md](architecture/execution_flow.md)，接著看 [training/](training/training_loop.md) |
| **4. 模型整合** | 模型如何接入 `BaseModel` 的契約 | [model/](model/model_architecture.md) |
| **5. 部署最佳化** | checkpoint 如何變成 TensorRT engine | [deployment/](deployment/export_pipeline.md) |

---

## 建議閱讀順序

1. [architecture/framework_overview.md](architecture/framework_overview.md) — 世界觀與 repository 地圖。**從這裡開始。**
2. [architecture/data_flow.md](architecture/data_flow.md) — 從頭到尾追蹤一筆 sample。
3. [architecture/execution_flow.md](architecture/execution_flow.md) — 執行 `autoware-ml train` 時究竟發生了什麼事。
4. [code_walkthrough/entry_point.md](code_walkthrough/entry_point.md) — 逐函式的實際程式碼追蹤。
5. [code_walkthrough/config_flow.md](code_walkthrough/config_flow.md) — Hydra config 如何組合而成。
6. [code_walkthrough/important_classes.md](code_walkthrough/important_classes.md) — 你會最常接觸到的類別，作為參考卡片。
7. 接著再依各個領域深入：[dataset/](dataset/dataset_pipeline.md) → [model/](model/model_architecture.md) → [training/](training/training_loop.md) → [evaluation/](evaluation/evaluation_pipeline.md) → [deployment/](deployment/export_pipeline.md)。

---

## 本指南的目錄結構

```text
docs/onboarding/
├── README.md                       ← you are here
├── architecture/
│   ├── framework_overview.md        Big picture, "why", stack comparison, repo map
│   ├── data_flow.md                 One sample's journey: info → batch → forward → loss
│   └── execution_flow.md            Runtime flow of `autoware-ml train`
├── dataset/
│   ├── dataset_pipeline.md          DataModule / Dataset / databases / collation
│   └── augmentation.md              Transforms library and the dict-in/dict-out contract
├── model/
│   ├── model_architecture.md        BaseModel contract + how a model is assembled
│   ├── backbone.md                  Backbones (SECOND, ResNet, sparse encoders, PTv3)
│   ├── neck.md                      Necks (SECONDFPN, LSS-FPN, CP-FPN)
│   └── head.md                      Heads (CenterHead, TransFusionHead, StreamPETRHead)
├── training/
│   ├── training_loop.md             Trainer, callbacks, the shared step, MLflow
│   ├── optimizer_scheduler.md       configure_optimizers, partials, custom schedulers
│   └── loss_design.md               Where losses live and how they are computed
├── evaluation/
│   ├── evaluation_pipeline.md       MetricSuite/Metric lifecycle, val vs test
│   └── metrics.md                   mAP, NDS, IoU, range-awareness
├── deployment/
│   ├── export_pipeline.md           build_export_spec → ONNX, multi-head exports
│   └── onnx_tensorRT.md             torch.onnx.export, TensorRT engine build, custom ops
└── code_walkthrough/
    ├── entry_point.md               console-script → main() → trainer.fit(), with file:line
    ├── config_flow.md               defaults → base → leaf, _target_, _partial_, interpolation
    └── important_classes.md         Reference card of the key classes
```

---

## 一段話重點整理（每次迷路時重讀一次）

Autoware-ML 是一個基於 **PyTorch Lightning + Hydra** 的框架。一份 **Hydra YAML config**
（`autoware_ml/configs/tasks/<task>/<model>/<variant>_<dataset>.yaml`）就能完整描述一次執行（run）。
CLI（`autoware-ml train|test|deploy`）會組合出這份 config，並對其中每個區塊呼叫
`hydra.utils.instantiate()`，藉此建立一個 **`DataModule`**、一個**模型**
（`BaseModel` 的子類別，而 `BaseModel` 本身*就是*一個 `LightningModule`）、Lightning 的
**callback**、一個 **MLflow logger**，以及一個 **`Trainer`** — 接著呼叫 `trainer.fit()`。
模型只需要實作兩個抽象方法：`forward()` 與 `compute_metrics()`（必須回傳一個
`"loss"`）；其餘一切 — 包括 training/val/test/predict 各步驟、optimizer 設定、
metric 紀錄，以及 ONNX 匯出 — 都是從 `BaseModel` 繼承而來。**幾乎每一個物件都是從
config 的 `_target_` 建構出來的，所以大多數的 bug 其實是 config 的問題，
修正的地方通常在 YAML，而不是 Python。**
