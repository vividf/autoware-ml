# Training Loop

> **What this covers:** how a batch becomes a weight update — the Lightning `Trainer`, the
> shared step, callbacks, precision/DDP/accumulation, MLflow logging, and resume/transfer.
> You never write a training loop here; you configure one.
>
> Prerequisites: [../architecture/execution_flow.md](../architecture/execution_flow.md),
> [../model/model_architecture.md](../model/model_architecture.md).

---

## 1. Who runs the loop

There is **no hand-written loop**. `scripts/train.py` builds a `lightning.Trainer` and calls
`trainer.fit(model, datamodule)`. Lightning then calls the hooks the model inherited from
`BaseModel`. Your job is to (a) implement `forward`/`compute_metrics` in the model and (b)
configure the trainer/callbacks/logger in YAML.

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

## 2. The Trainer config (`configs/defaults/modules/trainer.yaml`)

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

Tasks override what they need in their `base.yaml`. Examples:

| Model | Overrides |
| ----- | --------- |
| CenterPoint | `max_epochs: 30`, `gradient_clip_val: 5.0`, `gradient_clip_algorithm: norm` |
| StreamPETR | `max_epochs: 35`, `precision: bf16-mixed`, `use_distributed_sampler: false`, `gradient_clip_val: 1.0` |
| FRNet | step-based validation: `val_check_interval: 1500` |

Everything is a standard Lightning `Trainer` argument, so the Lightning docs apply directly.
The trainer is instantiated by `instantiate_trainer` (`utils/runtime.py`), which injects
`callbacks`, `logger`, and `default_root_dir` in code:

```python
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks,
                                  logger=trainer_logger or False, default_root_dir=root_dir)
```

### Precision, DDP, accumulation (all config)

| Want | Set |
| ---- | --- |
| Mixed precision | `trainer.precision=16-mixed` (or `bf16-mixed`) |
| Multi-GPU | `trainer.devices=[0,1]` (DDP auto-selected) or `trainer.devices=4 trainer.strategy=ddp` |
| Gradient accumulation | `trainer.accumulate_grad_batches=4` |
| Gradient clipping | `trainer.gradient_clip_val=5.0 trainer.gradient_clip_algorithm=norm` |

Torch runtime is set once in `configure_torch_runtime()` (`utils/runtime.py`): TF32 matmul
(`set_float32_matmul_precision("medium")`) + cuDNN TF32. Seeding is
`L.seed_everything(cfg.seed, workers=True)`.

---

## 3. The shared step (recap, `models/base.py:239`)

Every train/val/test batch runs the same core (`_shared_step`): filter batch → `forward` →
`compute_metrics` → assert `"loss"` → log. The per-mode wrappers differ only in prefix and
what they return:

```python
training_step:   metrics, _ = _shared_step(batch, "train", on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                 return metrics["loss"]                       # Lightning back-props this
validation_step: metrics, outputs = _shared_step(batch, "val", ...)
                 return {**metrics, "model_outputs": outputs}  # outputs kept for metric suites
```

So **losses are logged as `train/loss`, `train/loss_heatmap`, … and `val/loss`, …** with
`sync_dist=True` (Lightning averages the scalar across GPUs). Epoch-level *metrics* (mAP, etc.)
are a separate path — see [../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md).

---

## 4. Callbacks (`configs/defaults/modules/callbacks.yaml`)

Four callbacks ship by default:

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

`instantiate_callbacks` (`utils/runtime.py`) instantiates each, **skips** `LearningRateMonitor`
when there's no logger, and rewrites `ModelCheckpoint.dirpath` to the MLflow-owned checkpoint
dir when logging is enabled (so checkpoints land in the run's artifact tree).

### The one custom callback: config-authoritative `EarlyStopping` (`callbacks/early_stopping.py`)

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

**Why it exists:** stock Lightning restores a callback's *entire* state on resume, silently
reverting config changes (e.g. you raise `patience` from 20→40, but the old 20 is restored).
This mixin keeps *configuration* keys (constructor args like `patience`) at their configured
values while restoring *runtime progress* (wait counters) — and logs every override. This
reflects the framework's "config is authoritative on resume" philosophy (the same idea appears
in the `param_drift` MLflow tag for hyperparameters).

`autoware_ml/callbacks/` contains *only* this — there is no EMA or visualization callback here.

---

## 5. Logging (MLflow)

`configs/defaults/modules/logger.yaml` → `lightning.pytorch.loggers.MLFlowLogger`
(`tracking_uri: sqlite:///mlruns/mlflow.db`), instantiated only when a logger is configured.
The CLI pre-creates the run (see [../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md)),
and `scripts/train.py` populates it:

- **Hyperparameters** — `log_hyperparameters(cfg, logger)` logs the fully-resolved config
  (`OmegaConf.to_container(cfg, resolve=True)`, sanitized). On resume, params are append-only
  and drift is recorded in a `param_drift` tag.
- **Config artifacts + run metadata** written to the artifact dir before training.
- **Metrics/losses** — flow through `self.log_dict` in the model to the attached logger.
  Loss keys `{split}/loss...`; metric keys `{split}/{suite_prefix}/{metric}` (e.g.
  `val/det3d/mAP`). Checkpoint monitors and Optuna targets point at these keys directly.

View it: `autoware-ml mlflow ui --port 5000`.

---

## 6. Resume vs transfer (`--resume-checkpoint` vs `--weights`)

These are mutually exclusive (enforced in the CLI and again in `scripts/train.py`):

| Flag | Restores | Use for |
| ---- | -------- | ------- |
| `--resume-checkpoint <last.ckpt>` | model **+ optimizer + epoch**; continues the source MLflow run | resuming an interrupted run (`trainer.fit(..., ckpt_path=...)`) |
| `--weights <ckpt>` (repeatable) | **model weights only** (`apply_matching_weights`, `strict=False`) | transfer learning / initializing an encoder from another checkpoint |

`--weights` can be passed multiple times; later checkpoints overwrite earlier ones on
overlapping keys (used for multi-head merges).

---

## 7. Debugging a training run

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

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Checkpoint never saved / wrong metric | `ModelCheckpoint.monitor` key not logged | ensure `val/loss` (or your key) is logged; check names |
| Early stopping fires too early after config change | (stock Lightning would) restore old `patience` | this framework keeps the **configured** value — check the override warning in logs |
| Loss is `nan` | LR too high, bad GT, fp16 overflow | `detect_anomaly=true`; try `bf16-mixed`; lower LR; clip grads |
| No validation running | `check_val_every_n_epoch` / `val_check_interval` | set them; ensure `val_dataloader` exists |
| Multi-GPU slower/hangs | wrong `strategy`/`devices`, or sampler | `devices=[0,1]`; some models set `use_distributed_sampler: false` |
| `optimized_metric was not logged` | monitored key absent | log it, or set `+optimized_metric=<a logged key>` |
| Metrics not appearing in MLflow | logger disabled or key mismatch | ensure `cfg.logger` present; check `{split}/{prefix}/{key}` |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Train longer / shorter | `trainer.max_epochs=N` |
| Speed up | `trainer.precision=16-mixed`; raise `num_workers`; `pin_memory` |
| Bigger effective batch | `trainer.accumulate_grad_batches=N` |
| Change checkpoint criterion | `callbacks.model_checkpoint.monitor=val/det3d/mAP mode=max` |
| Disable early stopping | remove/override the `early_stopping` callback |
| Add a callback | add an entry under `callbacks:` with its `_target_` |
| Custom callback behavior on resume | subclass with `ConfigAuthoritativeStateMixin` if config should win |

---

**Next:** [optimizer_scheduler.md](optimizer_scheduler.md) · [loss_design.md](loss_design.md).
