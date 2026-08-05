# 程式碼逐步解析 — 進入點 (Entry Point)

> 這是 `autoware-ml train --config-name ...` 從 shell 到 `trainer.fit()` 逐一函式的實際追蹤過程。請在閱讀本文件的同時開啟對應的檔案並跟著操作。概念版請參閱 [../architecture/execution_flow.md](../architecture/execution_flow.md)；本文件則是「閱讀實際程式碼」的版本。

涉及的檔案（依呼叫順序排列）：

```text
pyproject.toml                       [project.scripts] entry
autoware_ml/cli/cli.py               Typer app + the `train` command
autoware_ml/utils/cli/helpers.py     run_lazy_script (lazy import + call)
autoware_ml/cli/runtime.py           run_hydra_entrypoint, prepare_runtime_environment
autoware_ml/scripts/train.py         the real @hydra.main entrypoint
autoware_ml/utils/runtime.py         instantiate_trainer / instantiate_callbacks / seed
```

---

## 步驟 0 — console script

`pyproject.toml`:

```toml
[project.scripts]
autoware-ml = "autoware_ml.cli.cli:main"
```

執行 `pip install -e .` 會建立一個 `autoware-ml` 執行檔，這個執行檔會呼叫
`autoware_ml.cli.cli:main`，進而執行 Typer app（`app()`）。Typer 會檢查 `argv[1]`
（也就是 `train`），並將呼叫分派到對應的指令函式。

為什麼用 Typer 而不是純 argparse：Typer 免費提供 shell 的 tab 自動完成與型別化選項，
而且 CLI 刻意保持匯入輕量，讓自動完成速度夠快（匯入時不會載入 torch/Hydra）。

---

## 步驟 1 — `train` 指令（`autoware_ml/cli/cli.py:205`）

```python
@app.command(
    name="train",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},  # ← let Hydra overrides through
)
def train(
    ctx: typer.Context,
    config_name: Annotated[str, typer.Option("--config-name", ...)],
    weights: Annotated[list[str] | None, typer.Option("--weights", ...)] = None,        # repeatable
    resume_checkpoint: Annotated[str | None, typer.Option("--resume-checkpoint", ...)] = None,
    new_run: Annotated[bool, typer.Option("--new-run", ...)] = False,
) -> None:
    if weights and resume_checkpoint:
        raise typer.BadParameter("--weights and --resume-checkpoint are mutually exclusive.")   # :264
    if new_run and not resume_checkpoint:
        raise typer.BadParameter("--new-run requires --resume-checkpoint.")                      # :266

    hydra_overrides: list[str] = []
    if weights:
        weights_list = "[" + ",".join(weights) + "]"
        hydra_overrides.append(f"+weights={weights_list}")            # :272  → cfg.weights
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise typer.BadParameter(...)                            # validated up front
        hydra_overrides.append(f"+resume_checkpoint={resume_checkpoint}")   # :278  → cfg.resume_checkpoint

    run_lazy_script(                                                 # :280
        CLI_RUNTIME_MODULE,                # "autoware_ml.cli.runtime"
        "run_hydra_entrypoint",
        entrypoint_module=TRAIN_ENTRYPOINT_MODULE,   # "autoware_ml.scripts.train"
        config_name=config_name,
        stage="train",
        extra_args=ctx.args,               # ← everything else, e.g. ["trainer.max_epochs=100"]
        hydra_overrides=hydra_overrides,
        resume_checkpoint=resume_checkpoint,
        new_run=new_run,
        config_prefix=TASK_CONFIG_PREFIX,  # "tasks"
    )
```

重點觀察：

- **`allow_extra_args` + `ignore_unknown_options`** 讓你可以在已知的旗標之後，附加原始的
  Hydra overrides（`trainer.max_epochs=100`、`model.optimizer.lr=1e-4`）。這些內容會以
  `ctx.args` 的形式傳入，並以 `extra_args` 轉發出去。
- **`--weights` / `--resume-checkpoint` 會變成 Hydra overrides**（`+weights=[...]`、
  `+resume_checkpoint=...`）。`+` 會*新增*一個頂層的 config 鍵。之後 `scripts/train.py`
  會用 `cfg.get("weights")` / `cfg.get("resume_checkpoint")` 讀取它們。
- **`+`** 很重要：它會新增一個不在 schema 中的鍵。覆寫*既有*的鍵則不需要 `+`。
  （請參閱 [config_flow.md](config_flow.md)。）
- `deploy` 與 `test` 的結構相同；只有 `stage=` 與 entrypoint module 不同（`deploy` 還會傳遞
  `checkpoints=weights`，用於多 checkpoint 的 MLflow lineage）。

`run_lazy_script`（`autoware_ml/utils/cli/helpers.py`）刻意寫得很簡單：

```python
def run_lazy_script(module_path, function_name, *args, **kwargs):
    module = importlib.import_module(module_path)   # torch/Hydra imported HERE, not at CLI startup
    return getattr(module, function_name)(*args, **kwargs)
```

---

## 步驟 2 — Hydra 橋接（`autoware_ml/cli/runtime.py:277`）

```python
def run_hydra_entrypoint(entrypoint_module, config_name, stage, extra_args=(), hydra_overrides=(), ...):
    env_updates = {}
    if stage is not None:
        env_updates = prepare_runtime_environment(config_name, config_prefix, stage, ...)   # :292

    sys.argv = resolve_hydra_entrypoint_argv(       # :304  build the argv @hydra.main will read
        entrypoint_module, config_name, config_prefix,
        extra_args=extra_args, hydra_overrides=hydra_overrides,
    )

    with (
        temporary_main_module(resolve_module_spec(entrypoint_module)),
        temporary_environment(env_updates),         # exports AUTOWARE_ML_RUN_ID / _HYDRA_RUN_DIR
    ):
        run_lazy_script(entrypoint_module, "main")   # :316  → scripts/train.py:main()
```

這個函式在呼叫真正的 entrypoint 之前，會做**兩件**事：

### 2a. 預先建立 MLflow run（`prepare_runtime_environment:188`）

```python
GlobalHydra.instance().clear()
with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
    cfg = compose(config_name=resolved_config_name, overrides=compose_overrides)   # :217  THROWAWAY compose

if should_enable_logger(cfg):                        # cfg.logger present?
    ...
    run_context = prepare_run_context(cfg.logger.tracking_uri, config_name, ...)   # :254  create MLflow run NOW
    return {
        AUTOWARE_ML_RUN_ID_ENV: run_context.run_id,          # :265
        AUTOWARE_ML_HYDRA_RUN_DIR_ENV: str(run_context.hydra_dir),
    }
return {AUTOWARE_ML_RUN_ID_ENV: None, AUTOWARE_ML_HYDRA_RUN_DIR_ENV: str(generate_hydra_run_dir(...))}
```

**為什麼要有一次用過即丟的 compose？** 為了讀取 `cfg.logger.tracking_uri`，並在真正的工作
*之前*建立 MLflow run，讓 run id 與目錄能事先確定。真正的工作之後會透過這兩個環境變數重複
使用它們。對 `deploy`/`test` 而言，這裡也是解析**run lineage**（parent/source run）的地方
（`resolve_deploy_lineage`、`resolve_lineage_context`）。

### 2b. 固定 run 目錄

`AUTOWARE_ML_HYDRA_RUN_DIR` 會被 `configs/defaults/modules/run.yaml` 使用：

```yaml
hydra:
  run:
    dir: ${oc.env:AUTOWARE_ML_HYDRA_RUN_DIR,mlruns/${user_config_name:${hydra:job.config_name}}/_hydra/...}
```

因此 Hydra 的輸出目錄會等於 MLflow 的 run 目錄 — checkpoint、config 快照、以及 Hydra 的
日誌都會放在同一個地方。`scripts/train.py:73` 會斷言（assert）兩者一致。

---

## 步驟 3 — 真正的 entrypoint（`autoware_ml/scripts/train.py:56`）

```python
_CONFIG_PATH = get_config_path()          # → str(CONFIGS_ROOT) == autoware_ml/configs

@hydra.main(version_base=None, config_path=_CONFIG_PATH)   # ← Hydra composes cfg HERE
def main(cfg: DictConfig):
    log_configuration(cfg)
    work_dir = resolve_work_dir()
    config_name = get_user_config_name()
    logger_enabled = should_enable_logger(cfg)
    if logger_enabled:
        pre_created_run_id = os.environ.get(AUTOWARE_ML_RUN_ID_ENV)   # set by the bridge in step 2
        if pre_created_run_id is not None:
            run_context = load_run_context(cfg.logger.tracking_uri, pre_created_run_id)
            if work_dir != run_context.hydra_dir:                     # :73  dirs must agree
                raise RuntimeError(...)
        else:
            run_context = prepare_run_context(...)                    # fallback: create it now

    configure_torch_runtime()          # :86  TF32 matmul, cudnn tf32
    set_seed(cfg)                      # :87  L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)              # :90  → a DataModule
    model      = hydra.utils.instantiate(cfg.model)                  # :93  → a BaseModel
    model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))   # :94

    # --weights / --resume-checkpoint (mutually exclusive)
    if weights_path is not None:  apply_matching_weights(model, weights_path, map_location="cpu", ...)   # :101
    if resume_checkpoint_path is not None:  ... # logs epoch/step it resumes from                        # :102

    callbacks      = instantiate_callbacks(cfg, logger_enabled=..., checkpoint_dir=...)   # :117
    trainer_logger = hydra.utils.instantiate(cfg.logger) if logger_enabled else None      # :138
    trainer        = instantiate_trainer(cfg, callbacks, trainer_logger, root_dir)        # :141

    log_hyperparameters(cfg, trainer_logger)                          # :148  MLflow params

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_checkpoint_path)   # :156  ← TRAINING

    score = trainer.callback_metrics.get(cfg.get("optimized_metric", "val/loss"))  # :166
    if score is None:  raise ValueError(...)                          # must have been logged
    return float(score)                                               # :173  for Optuna
```

**最需要牢記在心的一行：** 每個主要物件都是透過 `hydra.utils.instantiate(cfg.<section>)`
建立的。`cfg.datamodule`、`cfg.model`、`cfg.logger`，以及（在 helper 內部的）`cfg.callbacks`
與 `cfg.trainer`，全部都只是 config tree；Hydra 會讀取每個 `_target_` 並建構對應的物件。這裡
沒有 registry 查詢機制。關於這些 tree 是如何組成的，請參閱 [config_flow.md](config_flow.md)。

`instantiate_trainer`（`utils/runtime.py`）是一個很薄的 wrapper：

```python
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=trainer_logger or False,
                                  default_root_dir=root_dir)   # cfg.trainer._target_ == lightning.Trainer
```

`instantiate_callbacks` 會遍歷 `cfg.callbacks.values()` 並逐一實例化，當沒有 logger 時會跳過
`LearningRateMonitor`，並把 `ModelCheckpoint.dirpath` 改寫成 MLflow 的 checkpoint 目錄。

---

## 步驟 4 — `trainer.fit()` 之後

從 `trainer.fit(...)` 開始，你就進入了 Lightning 的世界。Lightning 會呼叫你的模型從
`BaseModel`（`models/base.py`）繼承而來的 hook：

```text
setup → configure_optimizers            (once)
per training batch:
    on_after_batch_transfer  →  training_step  →  loss.backward()  →  optimizer.step()
per validation epoch:
    validation_step ×N  →  on_validation_epoch_end (metric suites)  →  ModelCheckpoint(monitor=val/loss)
```

這些 hook 的詳細說明請參閱 [important_classes.md](important_classes.md)（BaseModel）與
[../training/training_loop.md](../training/training_loop.md)。

---

## 整條鏈路的濃縮版

```text
autoware-ml train --config-name detection3d/centerpoint/voxel020_..._nuscenes trainer.max_epochs=50
  │
  ▼ pyproject [project.scripts]
autoware_ml.cli.cli:main()  →  app()  →  train() command        cli.py:210
  │   builds hydra_overrides (+weights / +resume_checkpoint); extra_args = ["trainer.max_epochs=50"]
  ▼ run_lazy_script("autoware_ml.cli.runtime", "run_hydra_entrypoint", ...)
run_hydra_entrypoint(...)                                        runtime.py:277
  ├─ prepare_runtime_environment()  → throwaway compose, create MLflow run, set env vars   runtime.py:188
  ├─ sys.argv = ["--config-name", "tasks/detection3d/...", "trainer.max_epochs=50", ...]
  ▼ run_lazy_script("autoware_ml.scripts.train", "main")
scripts/train.py:main(cfg)   @hydra.main composes cfg           train.py:56
  ├─ instantiate datamodule / model / callbacks / logger / trainer
  ▼ trainer.fit(model, datamodule=datamodule, ckpt_path=...)    train.py:156
Lightning loop → BaseModel hooks (models/base.py)
```

---

## 常見除錯情境

| 症狀 | 問題出在哪裡 | 該檢查什麼 |
| ------- | ------------------ | ------------- |
| 未知選項／參數被拒絕 | `train` 指令的簽章 | 這是型別化的旗標；原始的 Hydra overrides 要放在已知旗標*之後*，以 `ctx.args` 的形式傳入 |
| `+weights`「could not override」 | 你在既有的鍵上使用了 `+`，或是在新鍵上省略了 `+` | 新鍵需要 `+`；既有的鍵則不需要 |
| 「Hydra work directory does not match…」 | `train.py:73` | `AUTOWARE_ML_HYDRA_RUN_DIR` 與 MLflow run 目錄不一致；通常是過期（stale）的環境變數 |
| MLflow run 已建立，但訓練始終沒有開始 | `prepare_runtime_environment` 中那次用過即丟的 compose 失敗了 | 用同樣的 config 加上 `--cfg job` 重新執行，以查看 composition 的錯誤 |
| `optimized_metric ... was not logged` | `train.py:167` | 被監控的 metric 鍵必須真的有被記錄（例如 `val/loss`） |
| 只有在 CLI 底下才會出現的怪異 import 錯誤 | 某個 module 在 CLI 的頂層被匯入（破壞了 lazy 設計） | 把重量級的 import 留在函式內部／`scripts/` 的 entrypoint 內 |

---

## 常見修改情境

| 我想要…… | 這樣做 |
| ---------- | ------- |
| 在 `train` 中新增一個 `--foo` 旗標 | 在 `train()` 中加入一個型別化的 `typer.Option`，將它轉換成 Hydra override，再於 `scripts/train.py` 中透過 `cfg.get("foo")` 讀取 |
| 在 `fit` 之後立即執行 `test` | 在 `scripts/train.py:main` 的最後加上 `trainer.test(...)` |
| 變更 run／artifact 的存放位置 | `configs/defaults/modules/run.yaml` + `utils/mlflow_helpers.py` |
| 新增一個指令（例如 `benchmark`） | 在 `cli.py` 中加入一個 `@app.command`，並在 `scripts/benchmark.py` 中撰寫一個 `main`，透過 `run_hydra_entrypoint` 或 `run_lazy_script` 分派 |

---

**下一篇：** [config_flow.md](config_flow.md) — 驅動這一切的 `cfg` 是如何組成的。
