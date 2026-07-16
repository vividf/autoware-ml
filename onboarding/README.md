# autoware-ml 新人上手筆記

> 這一份筆記是寫給「已經熟悉 OpenMMLab（mmdetection / mmdetection3d）與 AWML，第一次接觸 `autoware-ml`」的人。
> 目標是讓你用「已經會的東西」去對照理解新框架，而不是從零開始讀官方文件。
>
> 官方英文文件在 [`docs/`](../docs/)（用 zensical 產生，會發佈到 <https://tier4.github.io/autoware-ml>）。
> 這份 `onboarding/` 是**額外的中文對照筆記**，不屬於官方文件、預設不會被 commit，你可以自由搬移或刪除。

---

## 目錄

| 檔案 | 內容 | 什麼時候看 |
| --- | --- | --- |
| **README.md**（本檔） | 這是什麼、為什麼存在、30 秒心智模型、現況 | 先看這個 |
| [`01-awml-vs-autoware-ml.md`](01-awml-vs-autoware-ml.md) | **核心**：為什麼要重寫、技術棧對照、**概念對照表（mmdet/AWML → autoware-ml）**、目錄結構對照 | 想知道「跟 AWML 差在哪」 |
| [`02-architecture.md`](02-architecture.md) | 架構深入：整條資料流、每個子系統（Hydra / DataModule / transforms / BaseModel / ops / metrics / deploy） | 想知道「內部怎麼運作」 |
| [`03-workflow-and-gotchas.md`](03-workflow-and-gotchas.md) | 日常流程（create-dataset → train → test → deploy）＋ 所有值得注意的坑 | 想開始動手做 |

---

## 一句話：`autoware-ml` 是什麼？

> **`autoware-ml` 是 TIER IV 對 AWML 的「去 OpenMMLab 化」重寫版本，改用 PyTorch Lightning + Hydra 當骨架，為 Autoware 的自動駕駛感知模型提供「訓練 → 評估 → 部署（ONNX/TensorRT）」的一條龍框架。**

它會**取代**現在的 [`tier4/AWML`](https://github.com/tier4/AWML)。目前是 **Early Alpha**。

- Distribution 名稱：`autoware-ml`（GitHub repo、`pip install autoware-ml`、docker image）
- Python package 名稱：`autoware_ml`（`import autoware_ml`）
- 對外品牌／文章寫法：Autoware-ML

> ⚠️ **命名陷阱**：AWML 那個舊 repo，它的 Python package **也叫 `autoware_ml/`**。兩個 repo 的頂層資料夾同名但內容完全不同。本筆記講的 `autoware_ml/` 一律指新 repo（`~/ml_workspace/autoware-ml`）。舊 repo（`~/ml_workspace/AWML`）裡的 `graphify-out/` 那份圖譜是描述**舊**框架的，不要拿來套新的。

---

## 為什麼要有這個專案？（你的理解是對的）

AWML 的定位是「基於 mmdetection3d 的 Autoware ML 框架」——但它其實是**把 mmdetection3d fork 過來、再貼上一層 T4/Autoware**。它硬綁死了 **7 個 OpenMMLab 套件**（`mmengine`、`mmcv`、`mmdet`、`mmdet3d`、`mmdeploy`、`mmpretrain`、`mmsegmentation`），而且：

- 用 OpenMMLab 自己的 `mim` 安裝，版本全部釘死（mmdet3d 1.4.0、mmcv 2.1.0…），連 PyTorch 都被卡在 2.x 舊版。
- 把 mmengine 的全域 registry 直接 re-export 當自己的（`autoware_ml/registry.py`），設 `default_scope = "mmdet3d"`。
- 連 `setup.py` 都還自稱是「OpenMMLab's next-generation platform for general 3D object detection」、作者掛「MMDetection3D Contributors」。
- 甚至要**在 Docker build 時 monkeypatch mmdet3d / mmengine / spconv 的 site-packages**，才能讓它在新版 numpy / PyTorch 上跑起來。

這帶來的問題正是你說的「太多依賴」：

1. **升級地獄**——想升 PyTorch / CUDA / numpy，就會被一整串 mm 套件的版本限制卡死。
2. **黑箱**——模型行為藏在 mmdet3d/mmengine 深處，除錯要跨三四個 repo。
3. **部署脆弱**——舊部署路徑綁 `mmdeploy`，客製 ONNX/TensorRT 很難維護。
4. **裝不起來**——依賴解析複雜、需要 patch。

`autoware-ml` 的解法就是**整個骨架重寫、完全不依賴任何 mm 套件**，換成社群標準、各自獨立維護的元件：

| 角色 | AWML（舊） | autoware-ml（新） |
| --- | --- | --- |
| 設定系統 | mmengine `Config`（`.py` dict + registry `type=`） | **Hydra + OmegaConf**（`.yaml` + `_target_`） |
| 訓練迴圈 | mmengine `Runner` | **PyTorch Lightning `Trainer`** |
| 模型基底 | mmdet3d 的 `MVXTwoStageDetector` / mmengine `BaseModule` | 自己的 `BaseModel`（繼承 `LightningModule`） |
| 元件註冊 | 全域 Registry + `@MODELS.register_module()` | **不需要**（Hydra 直接用 Python 路徑 import） |
| 低階 ops | `mmcv` ops（voxelization、spconv…） | **自帶原生 `autoware_ml/ops/`** |
| 部署 | `mmdeploy` | **`torch.onnx.export` + TensorRT Builder API** |
| 環境管理 | `mim` + Docker + patch | **`pixi`**（lockfile）+ Docker |
| 實驗追蹤 | （text log / work_dirs） | **MLflow + Optuna** |

> 細節與完整概念對照表在 [`01-awml-vs-autoware-ml.md`](01-awml-vs-autoware-ml.md)。

---

## 30 秒心智模型

把它想成四層，全部由 Hydra 的一份 YAML 組裝起來：

```
             ┌─────────────────────────────────────────────┐
  YAML 設定  │  Hydra 讀 config，把每個 _target_ 節點        │
  + Optuna   │  instantiate() 成真正的 Python 物件           │
             └───────────────┬─────────────────────────────┘
                             │  組出這四個物件：
        ┌────────────────────┼───────────────────────┬──────────────┐
        ▼                    ▼                       ▼              ▼
  LightningDataModule   BaseModel(LightningModule)  Trainer       (deploy)
  ├ Dataset             ├ 子模組（encoder/head…）    ├ Callbacks    build_export_spec
  ├ transforms（CPU）    ├ forward()                 ├ MLflowLogger  → ONNX
  └ collate_fn          ├ compute_metrics()→loss    └ Checkpoints   → TensorRT
                        └ on_after_batch_transfer
                          （DataPreprocessing，GPU）
```

一次訓練的資料流：

```
get_data_info()（只回傳 metadata）
   → transforms（CPU，讀檔＋擴增，dict-in/dict-out）
   → collate_fn（依 collation_map 把 batch 疊起來）
   → on_after_batch_transfer（DataPreprocessing，在 GPU 上，例如 voxelization）
   → forward()（用 inspect.signature 只餵對得上名字的 batch key）
   → compute_metrics()（回傳含 "loss" 的 dict）／ metrics suite 累積評估
```

指令長這樣（統一的 `autoware-ml` CLI 取代了 AWML 的 `tools/*/train.py`）：

```bash
autoware-ml create-dataset --dataset nuscenes --task segmentation3d --root-path ... --out-dir ...
autoware-ml train   --config-name segmentation3d/ptv3/voxel005_51m_nuscenes
autoware-ml test    --config-name ... --weights .../best.ckpt
autoware-ml deploy  --config-name ... --weights .../best.ckpt
autoware-ml mlflow ui
```

---

## 現在支援哪些模型（Early Alpha 現況）

| 任務 | Modality | 模型 | 訓練 | ONNX | TensorRT |
| --- | --- | --- | --- | --- | --- |
| Calibration Status（分類） | Camera+LiDAR | ResNet18 | ✅ | ✅ | ✅ |
| Detection 3D | LiDAR | CenterPoint | ⏳ 進行中（目前只有 head） | ⏳ | ⏳ |
| Detection 3D | LiDAR | TransFusion | ✅ | ✅ | ⏳ |
| Detection 3D | Camera+LiDAR | BEVFusion | ✅ | ✅ | ⏳ |
| Detection 3D | Camera | StreamPETR | ⏳ | ⏳ | ⏳ |
| Segmentation 3D | LiDAR | FRNet | ✅ | ✅ | ✅ |
| Seg 3D + Det 3D（多任務） | LiDAR | PointTransformerV3 | ✅ | ✅ | ⏳ |

**AWML 有、但目前還沒搬到 autoware-ml 的東西**（重要，別以為不見了）：

- 2D 相關：YOLOX / YOLOX_opt、GLIP、MobileNetv2 分類、BLIP-2、traffic light 相關
- Active learning 全套：auto-labeling（pseudo_label）、scene_selector、data mining
- WebAuto pipeline 整合
- 舊有的 `pipelines/`、大部分 `tools/` 分析工具

> autoware-ml 目前**聚焦在 3D 感知（detection / segmentation）的 train→deploy 主幹**。要用 active learning / 2D / auto-labeling，現階段還是得回 AWML。

---

## 下一步

1. 想知道「跟 AWML 到底差在哪、我腦中的 mmdet 概念怎麼對應」→ 看 [`01-awml-vs-autoware-ml.md`](01-awml-vs-autoware-ml.md)
2. 想知道「內部每個模組怎麼運作」→ 看 [`02-architecture.md`](02-architecture.md)
3. 想直接動手 train / deploy，並避開所有坑 → 看 [`03-workflow-and-gotchas.md`](03-workflow-and-gotchas.md)
