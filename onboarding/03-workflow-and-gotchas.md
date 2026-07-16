# 03 · 日常流程與值得注意的坑

- [安裝與環境](#安裝與環境)
- [完整工作流](#完整工作流)
- [config 撰寫 pattern](#config-撰寫-pattern)
- [除錯技巧](#除錯技巧)
- [值得注意的坑（gotchas）](#值得注意的坑gotchas)
- [新增一個模型的最短路徑](#新增一個模型的最短路徑)

---

## 安裝與環境

**硬體/驅動需求**：NVIDIA GPU（Compute Capability 8.0+）、driver ≥ 570（建議 580）、CUDA 12.8 對應環境。

**兩種安裝路線**：

- **Docker（Early Alpha 建議）**：
  ```bash
  git clone https://github.com/tier4/autoware-ml.git && cd autoware-ml
  docker pull ghcr.io/tier4/autoware-ml:latest      # 或 ./docker/build.sh 自己 build
  ./docker/container.sh --run --data-path /path/to/your/datasets
  ```
  image 內是用 lockfile 建好的完整 `dev` pixi 環境（Ubuntu 24.04 CUDA/cuDNN base）。**PyTorch 等整套 ML stack 全來自 lockfile，不是預載的 PyTorch image**。

- **本機 pixi**（要先自備 NVIDIA driver + CUDA toolkit `nvcc`，因為要編 ops）：
  ```bash
  pixi install --locked --environment dev
  pixi run --environment dev setup-project     # 編 ops + 裝 bash completion
  pixi shell --environment dev
  ```
  環境分三個：`default`（只跑）、`dev`（貢獻者：含 compilers/tmux/docs 工具，**編 ops 必用**）、`docs`（只建文件，CI 用）。

**資料路徑**：所有 dataset 建議放同一個目錄，用腳本設環境變數：
```bash
./set_data_path.sh /path/to/your/datasets   # 設 AUTOWARE_ML_DATA_PATH
source ~/.bashrc
```
`docker/container.sh`、`.devcontainer`、model config 都靠這個變數定位資料。

> 對照 AWML：`mim install` 7 個 mm 套件 + patch → 換成 `pixi install --locked`。`work_dirs/` → `mlruns/`。

---

## 完整工作流

以官方 quickstart 的 PTv3 nuScenes 分割為例：

```bash
# 0. 進容器
./docker/container.sh --run

# 1. 產生 info 檔（取代 AWML 的 create_data_t4dataset.py）
autoware-ml create-dataset \
    --dataset nuscenes --task segmentation3d \
    --root-path data/nuscenes --out-dir data/nuscenes/info \
    --version v1.0-trainval

# 2. 訓練（config 名相對於 configs/tasks/，不用寫 tasks/ 前綴）
autoware-ml train --config-name segmentation3d/ptv3/voxel005_51m_nuscenes

# 3. 看 MLflow（loss 曲線、指標、超參）
autoware-ml mlflow ui --port 5000        # http://localhost:5000

# 4. 評估某個 checkpoint
autoware-ml test \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt

# 5. 匯出（TensorRT 對 PTv3 還沒好，先關掉只出 ONNX）
autoware-ml deploy \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights .../checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

**長時間訓練**：用內建 session（tmux-backed，取代 `nohup`）：

```bash
autoware-ml session start --name ptv3-train --cwd /workspace -- \
    train --config-name segmentation3d/ptv3/voxel005_51m_nuscenes
autoware-ml session attach --name ptv3-train    # 唯讀檢視，Ctrl+C 離開但不停訓練
autoware-ml session ls
autoware-ml session stop  --name ptv3-train
```

**resume / 遷移學習**（兩者互斥）：

```bash
# resume：續跑同一個 MLflow run（權重+optimizer+epoch 全還原）
autoware-ml train --config-name <cfg> \
    --resume-checkpoint .../checkpoints/last.ckpt

# --weights：只拿權重初始化（遷移學習），可重複、後蓋前
autoware-ml train --config-name <cfg> \
    --weights .../some_pretrained.ckpt
```

---

## config 撰寫 pattern

```yaml
# @package _global_                        # task config 幾乎都要這行
defaults:
  - /tasks/<task>/<model>/base             # 繼承 base
  - /datasets/nuscenes/detection3d         # 疊 dataset 片段
  - _self_                                 # 自己的覆寫最後套用（順序重要！）

data_root: ${oc.env:AUTOWARE_ML_DATA_PATH}/nuscenes

model:
  _target_: autoware_ml.models.detection3d.transfusion.TransFusionDetectionModel  # 完整路徑
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true                        # optimizer/scheduler 一定要 partial
    lr: 0.0001
  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}           # 插值引用別處

datamodule:
  collation_map:                           # 白名單！沒列的 key 會被丟掉
    points: list
    gt_boxes: list
    gt_labels: list
```

**CLI override**（Hydra 語法）：

```bash
# 改「已存在」的值：直接寫路徑（不要加 +）
autoware-ml train --config-name <cfg> trainer.max_epochs=100 model.optimizer.lr=5e-4

# 新增「原本沒有」的 key：要加 +
autoware-ml train --config-name <cfg> +trainer.fast_dev_run=true

# 掃參（multirun）
autoware-ml train --config-name <cfg> --multirun model.optimizer.lr=1e-3,5e-4,1e-4
```

命名慣例：`<task>/<model>/<variant>_<dataset>`，例如 `detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes`。voxel 寫成 `voxel005`，範圍寫成 `54m`、`122m`。

---

## 除錯技巧

```bash
# 只印組好的 config、不執行
autoware-ml train --config-name <cfg> --cfg job
autoware-ml train --config-name <cfg> --cfg job --package model   # 只看 model 段

# 跑一個 batch 驗證 pipeline 通不通
autoware-ml train --config-name <cfg> +trainer.fast_dev_run=true

# 限制 batch 數
autoware-ml train --config-name <cfg> +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

# 抓 NaN/inf 來源
autoware-ml train --config-name <cfg> +trainer.detect_anomaly=true
```

---

## 值得注意的坑（gotchas）

從 mmdet3d/AWML 過來最容易踩的，依重要性排：

1. **`forward()` 參數名 = batch key**。`BaseModel` 用 `inspect.signature(self.forward)` 只餵名字對得上的 key。`forward(self, points, gt_boxes)` 就得確保 batch（經 collation + preprocessing 後）有 `points`、`gt_boxes`。名字不合 → 該參數拿不到值。特殊批次/匯出需求請 override hook，不要繞過 `BaseModel`。

2. **`collation_map` 是白名單，沒列到的 key 會被丟掉**。你在 transform 產生的欄位，如果沒在 `collation_map` 指定策略，進不到 model。不定長點雲要用 `concat`/`index_concat`（會自動產生 `offset`），固定形狀用 `stack`，其他用 `list`。

3. **optimizer / scheduler 一定要 `_partial_: true`**。否則 Hydra 會在還沒有 `params` 時就直接呼叫 `AdamW()` 而爆掉。它們是「等拿到 model 參數再呼叫」的 factory。

4. **沒有 registry，import 路徑就是一切**。`_target_` 要指到**實作模組的完整路徑**（`autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile`），**不要靠 `__init__.py` 的 re-export**（框架刻意避免 package 層 re-export）。打錯路徑不會像 registry 那樣給你「KeyError: not registered」，而是 import error。

5. **`model.metrics` 是 list，Hydra 整包取代、不合併**。想微調指標範圍，不要重寫整個 suite，而是覆寫 `metric_ranges` / `metric_eval_class_range` 這兩個插值變數（suite 在 base config 定義一次，讀這兩個變數）。多任務串多個 suite 要用自訂 resolver `merge_lists`（`[${a}, ${b}]` 直接寫會壞）。

6. **`autoware-ml test` 預設單卡**。這是刻意的：多卡 DDP 的 validation sampler 會 pad 最後一個 batch（重複 frame），對指標有微小污染。要多卡評估才加 `--use-config-devices`。訓練途中的 val 本來就多卡，屬已知、可接受的誤差。

7. **輸出側後處理放在 model 內，不是 config pipeline**。logits→機率、voxel-to-point scatter、框解碼，一律寫在 `forward` / `compute_metrics` / `predict_outputs`。`data_preprocessing` pipeline **只做輸入側**（voxelization 等），且在 GPU 上（`on_after_batch_transfer`）。

8. **改 ops 要用 dev 環境重編**。`pixi run --environment dev setup-project`。`default` 環境沒有 compiler，編不動 `bev_pool_ext`。改完 `.cu`/`.cpp` 要重跑這個才會生效。

9. **`deploy` / `test` 的 `--weights` 強制全參數覆蓋**。多個 `--weights` 依序合併、後蓋前；載完若有參數沒被任何 checkpoint 覆蓋到，指令直接失敗並列出缺哪些 key（防止匯出含未訓練層）。多 head 模型（PTv3 det）常要餵兩個 checkpoint（backbone + head）。

10. **`databases/`（Polars/parquet）是 transitional**。程式碼註解明說 `generate_dataset.py` 之後會併回正式框架。目前 detection3d / segmentation3d 主要走 `datamodule/` 的 info-pkl 系統；`databases/` 主要服務 `MultiTaskDataModule`。別預期它的 API 穩定。

11. **命名撞名**：舊 repo AWML 的 package 也叫 `autoware_ml/`，且舊 repo 有 `graphify-out/` 圖譜——**那份圖譜描述的是舊框架**，不要拿來理解新 repo。本 repo（`~/ml_workspace/autoware-ml`）沒有 graphify 輸出。

12. **Early Alpha，會變**：CenterPoint 目前只有 head（沒有頂層模型/部署）；TransFusion / BEVFusion / PTv3 的 TensorRT 還在進行中（先 `deploy.tensorrt.enabled=false` 出 ONNX）；2D、auto-labeling、active learning、WebAuto 都還沒搬。要這些先回 AWML。

13. **`create-dataset` 目前 generator 只註冊 nuScenes**。T4dataset 的完整 info 生成路徑仍在補（部分走 `databases/` 的 parquet 系統）。

14. **`config-name` 相對於 `configs/tasks/`**。寫 `detection3d/transfusion/xxx`，不要寫成 `tasks/detection3d/...` 或絕對路徑。

15. **Docker 訓練凍結 + 容器內 `nvidia-smi` 失敗（`Failed to initialize NVML`）**：這是 NVIDIA Container Toolkit 在 `systemd` cgroup driver 下的已知問題，不是 model 的錯。重建容器（`./docker/container.sh --stop && --run`）；根治是把 Docker 改用 `cgroupfs`（見 `docs/framework/troubleshooting.md`）。

---

## 新增一個模型的最短路徑

（對照 AWML「開一個 `projects/<Model>/`、寫 mmengine config、`@register_module`、`custom_imports`」——新框架簡單很多）

1. **寫 model 類別**：`autoware_ml/models/<task>/<model>.py`，繼承 `BaseModel`，實作 `forward(**kwargs)` 與 `compute_metrics(batch, outputs) → {"loss": ...}`。需要客製就 override hook（`set_data_preprocessing` / `predict_outputs` / `get_log_batch_size` / `build_export_spec` / `build_eval_output`），**不要另外自己開 `LightningModule`**。
2. **需要新資料就寫 DataModule**：`autoware_ml/datamodule/<dataset>/<task>.py`，`Dataset` 實作 `get_data_info(index)→dict`（只回 metadata），`DataModule` 實作 `_create_dataset(split, transforms)`。
3. **需要新 transform**：`autoware_ml/transforms/<group>/<name>.py`，繼承 `BaseTransform`，實作 `transform(input_dict)→dict`。
4. **需要新 GPU 前處理**：`autoware_ml/preprocessing/<group>/<name>.py`，一個吃 `dict` 回 `dict` 的 callable。
5. **寫 config**：`autoware_ml/configs/tasks/<task>/<model>/base.yaml`（模型骨架）+ `<variant>_<dataset>.yaml`（疊 dataset 片段、填 `point_cloud_range` 等）。全部用 `_target_` 指到你剛寫的類別；optimizer/scheduler 記得 `_partial_: true`。
6. **train / deploy**：
   ```bash
   autoware-ml train  --config-name <task>/<model>/<variant>_<dataset>
   autoware-ml deploy --config-name <task>/<model>/<variant>_<dataset> --weights .../last.ckpt
   ```

不需要：註冊到任何 registry、設 scope、寫 `custom_imports`、開 per-project Dockerfile/setup.py。

> 官方英文版對照：`docs/contributing/adding-models.md`、`docs/framework/design.md`。
