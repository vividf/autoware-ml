# Training Loop

> **本文涵蓋內容：** 一個 batch 如何變成一次權重更新 — Lightning 的 `Trainer`、
> 共用的 step、callback、precision/DDP/累積（accumulation）、MLflow 記錄，以及
> resume/transfer。你在這裡永遠不需要自己寫 training loop，你只需要設定（configure）它。
>
> 先備知識：[../architecture/execution_flow.md](../architecture/execution_flow.md)、
> [../model/model_architecture.md](../model/model_architecture.md)。

---

## 1. 誰在執行這個 loop

**沒有手寫的 loop**。`scripts/train.py` 建構一個 `lightning.Trainer` 並呼叫
`trainer.fit(model, datamodule)`。接著 Lightning 會呼叫模型從 `BaseModel` 繼承而來的
各個 hook。你的工作是 (a) 在模型中實作 `forward`/`compute_metrics`，以及 (b) 在 YAML
中設定 trainer/callback/logger。

```mermaid
sequenceDiagram
    participant T as lightning.Trainer
    participant M as BaseModel
    participant C as Callbacks
    T->>M: configure_optimizers() (once)
    loop each training batch
        T->>M: on_after_batch_transfer(batch)  (GPU preprocessing)
        T->>M: training_step → _shared_step → forward → compute_metrics
        M-->>T: loss
        T->>T: loss.backward(); optimizer.step(); scheduler.step()
        T->>C: LearningRateMonitor logs lr
    end
    loop each validation epoch
        T->>M: validation_step ×N  (stashes model_outputs)
        T->>M: on_validation_epoch_end → metric suites compute
        T->>C: ModelCheckpoint(monitor=val/loss); EarlyStopping
    end
```

---

## 2. Trainer 的 config（`configs/defaults/modules/trainer.yaml`）

```yaml
# @package _global_
trainer:
  _target_: lightning.Trainer
  max_epochs: 10
  accelerator: gpu
  strategy: auto          # Lightning picks DDP automatically when devices > 1
  devices: auto           # all visible GPUs
  precision: 32-true      # override to 16-mixed / bf16-mixed for speed
  log_every_n_steps: 50
  val_check_interval: 1.0
  check_val_every_n_epoch: 1
  accumulate_grad_batches: 1
  enable_progress_bar: true
  enable_model_summary: true
```

各任務會在自己的 `base.yaml` 中覆寫（override）需要的部分。範例：

| 模型 | 覆寫 |
| ----- | --------- |
| CenterPoint | `max_epochs: 30`, `gradient_clip_val: 5.0`, `gradient_clip_algorithm: norm` |
| StreamPETR | `max_epochs: 35`, `precision: bf16-mixed`, `use_distributed_sampler: false`, `gradient_clip_val: 1.0` |
| FRNet | step-based validation: `val_check_interval: 1500` |

一切都是標準的 Lightning `Trainer` 參數，所以 Lightning 的官方文件可以直接套用。
trainer 是由 `instantiate_trainer`（`utils/runtime.py`）實例化（instantiate）的，該函式
會在程式碼中注入 `callbacks`、`logger`、`default_root_dir`：

```python
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks,
                                  logger=trainer_logger or False, default_root_dir=root_dir)
```

### Precision、DDP、累積（accumulation）（全部都是 config）

| 需求 | 設定 |
| ---- | --- |
| Mixed precision | `trainer.precision=16-mixed` (or `bf16-mixed`) |
| Multi-GPU | `trainer.devices=[0,1]` (DDP auto-selected) or `trainer.devices=4 trainer.strategy=ddp` |
| Gradient accumulation | `trainer.accumulate_grad_batches=4` |
| Gradient clipping | `trainer.gradient_clip_val=5.0 trainer.gradient_clip_algorithm=norm` |

Torch 執行環境（runtime）只會在 `configure_torch_runtime()`（`utils/runtime.py`）中設定一次：
TF32 matmul（`set_float32_matmul_precision("medium")`）+ cuDNN TF32。種子（seed）設定則是
`L.seed_everything(cfg.seed, workers=True)`。

---

## 3. 共用的 step（回顧，`models/base.py:239`）

每一個 train/val/test batch 都執行相同的核心流程（`_shared_step`）：過濾 batch →
`forward` → `compute_metrics` → 檢查（assert）`"loss"` 存在 → 記錄。各模式（mode）的
包裝函式僅在前綴（prefix）和回傳內容上有所不同：

```python
training_step:   metrics, _ = _shared_step(batch, "train", on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                 return metrics["loss"]                       # Lightning back-props this
validation_step: metrics, outputs = _shared_step(batch, "val", ...)
                 return {**metrics, "model_outputs": outputs}  # outputs kept for metric suites
```

因此**loss 會被記錄為 `train/loss`、`train/loss_heatmap`、… 以及 `val/loss`、…**，
並帶有 `sync_dist=True`（Lightning 會在各 GPU 間對該純量取平均）。Epoch 層級的
*metric*（mAP 等）走的是另一條路徑 — 參見
[../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md)。

---

## 4. Callback（`configs/defaults/modules/callbacks.yaml`）

預設會啟用四個 callback：

```yaml
callbacks:
  model_checkpoint:            # keeps the BEST by val/loss
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: val/loss
    dirpath: ${hydra:run.dir}/checkpoints
    filename: best
    save_top_k: 1
    mode: min
  model_checkpoint_last:       # always keeps last.ckpt (for resume)
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${hydra:run.dir}/checkpoints
    filename: last
    save_top_k: 1
    enable_version_counter: false
  early_stopping:              # CUSTOM (see below)
    _target_: autoware_ml.callbacks.early_stopping.EarlyStopping
    monitor: val/loss
    patience: 20
    mode: min
  lr_monitor:
    _target_: lightning.pytorch.callbacks.LearningRateMonitor
    logging_interval: step
```

`instantiate_callbacks`（`utils/runtime.py`）會實例化每一個 callback，在沒有 logger 時
**跳過** `LearningRateMonitor`，並在啟用 logging 時將 `ModelCheckpoint.dirpath` 重寫為
MLflow 所擁有的 checkpoint 目錄（讓 checkpoint 落在該次 run 的 artifact 樹狀結構中）。

### 唯一的自訂 callback：以 config 為準（config-authoritative）的 `EarlyStopping`（`callbacks/early_stopping.py`）

```python
class ConfigAuthoritativeStateMixin:
    def load_state_dict(self, state_dict):
        # any state key that is ALSO a constructor arg = configuration → the configured value wins
        config_keys = state_dict.keys() & inspect.signature(type(self).__init__).parameters.keys()
        state = dict(state_dict)
        for key in sorted(config_keys):
            if state[key] != getattr(self, key):
                logger.warning("%s.%s: checkpoint value %r overridden by configured value %r.", ...)
            state[key] = getattr(self, key)      # keep configured value, not checkpoint value
        super().load_state_dict(state)

class EarlyStopping(ConfigAuthoritativeStateMixin, LightningEarlyStopping): ...
```

**為什麼需要它：** 原生的 Lightning 會在 resume 時還原 callback 的*整個*狀態，
在你不知情的情況下悄悄地把 config 的更動蓋回去（例如你把 `patience` 從 20 提高到 40，
結果舊的 20 又被還原了）。這個 mixin 會讓*配置類*的 key（也就是 constructor 參數，
像 `patience`）維持在你所配置的值，同時還原*執行期進度*（等待計數器），並且會記錄下
每一次的覆寫。這反映了此框架「resume 時以 config 為準」的哲學（同樣的想法也出現在
超參數的 `param_drift` MLflow tag 中）。

`autoware_ml/callbacks/` 裡*只有*這一個東西 — 這裡沒有 EMA 或視覺化（visualization）callback。

---

## 5. Logging（MLflow）

`configs/defaults/modules/logger.yaml` → `lightning.pytorch.loggers.MLFlowLogger`
（`tracking_uri: sqlite:///mlruns/mlflow.db`），只有在設定了 logger 時才會被實例化。
CLI 會預先建立該次 run（參見
[../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md)），而
`scripts/train.py` 負責填入內容：

- **超參數（Hyperparameters）** — `log_hyperparameters(cfg, logger)` 會記錄完整解析
  （resolve）後的 config（`OmegaConf.to_container(cfg, resolve=True)`，經過消毒
  sanitize）。在 resume 時，參數是只能附加（append-only）的，任何漂移（drift）都會被
  記錄在 `param_drift` tag 中。
- **Config artifact + run 中繼資料（metadata）** 會在訓練開始前寫入 artifact 目錄。
- **Metrics/loss** — 透過模型中的 `self.log_dict` 流向所接上的 logger。
  Loss 的 key 是 `{split}/loss...`；metric 的 key 是 `{split}/{suite_prefix}/{metric}`
  （例如 `val/det3d/mAP`）。checkpoint 的 monitor 和 Optuna 的目標都是直接指向這些 key。

檢視方式：`autoware-ml mlflow ui --port 5000`。

---

## 6. Resume 與 transfer（`--resume-checkpoint` 對比 `--weights`）

這兩者互斥（在 CLI 及 `scripts/train.py` 中都有再次強制檢查）：

| 旗標 | 還原 | 用於 |
| ---- | -------- | ------- |
| `--resume-checkpoint <last.ckpt>` | model **+ optimizer + epoch**; continues the source MLflow run | resuming an interrupted run (`trainer.fit(..., ckpt_path=...)`) |
| `--weights <ckpt>` (repeatable) | **model weights only** (`apply_matching_weights`, `strict=False`) | transfer learning / initializing an encoder from another checkpoint |

`--weights` 可以傳入多次；在有重疊的 key 上，較後面的 checkpoint 會覆蓋較前面的
（用於多 head 合併的情境）。

---

## 7. 除錯一次訓練執行

```bash
# single batch, full train/val cycle
autoware-ml train --config-name <cfg> +trainer.fast_dev_run=true
# limit batches
autoware-ml train --config-name <cfg> +trainer.limit_train_batches=10 +trainer.limit_val_batches=5
# NaN hunting
autoware-ml train --config-name <cfg> +trainer.detect_anomaly=true
# see the exact composed config without running
autoware-ml train --config-name <cfg> --cfg job
```

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| Checkpoint 從未被儲存 / 監控錯誤的 metric | `ModelCheckpoint.monitor` 指定的 key 未被記錄 | 確認 `val/loss`（或你指定的 key）有被記錄；檢查名稱是否正確 |
| config 更動後 early stopping 太早觸發 | （原生 Lightning 會）還原舊的 `patience` | 此框架會保留**已配置**的值 — 檢查 log 中的覆寫警告 |
| Loss 是 `nan` | LR 太高、GT 有問題、fp16 溢位（overflow） | 使用 `detect_anomaly=true`；改用 `bf16-mixed`；降低 LR；裁剪梯度 |
| 沒有執行驗證（validation） | `check_val_every_n_epoch` / `val_check_interval` | 設定它們；確認 `val_dataloader` 存在 |
| 多 GPU 變慢或卡住 | `strategy`/`devices` 設定錯誤，或 sampler 問題 | 使用 `devices=[0,1]`；某些模型會設定 `use_distributed_sampler: false` |
| `optimized_metric was not logged` | 被監控的 key 不存在 | 記錄該 key，或設定 `+optimized_metric=<a logged key>` |
| Metrics 沒有出現在 MLflow 中 | logger 未啟用，或 key 不匹配 | 確認 `cfg.logger` 存在；檢查 `{split}/{prefix}/{key}` |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 訓練更久/更短 | `trainer.max_epochs=N` |
| 加速訓練 | `trainer.precision=16-mixed`；調高 `num_workers`；使用 `pin_memory` |
| 更大的有效 batch | `trainer.accumulate_grad_batches=N` |
| 更改 checkpoint 的判斷標準 | `callbacks.model_checkpoint.monitor=val/det3d/mAP mode=max` |
| 停用 early stopping | 移除或覆寫 `early_stopping` callback |
| 新增一個 callback | 在 `callbacks:` 底下新增一個帶有 `_target_` 的項目 |
| resume 時自訂 callback 行為 | 若希望 config 優先，繼承 `ConfigAuthoritativeStateMixin` |

---

**Next:** [optimizer_scheduler.md](optimizer_scheduler.md) · [loss_design.md](loss_design.md)。
