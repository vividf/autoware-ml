# Code Walkthrough — Entry Point

> The literal, function-by-function trace of `autoware-ml train --config-name ...`, from the
> shell to `trainer.fit()`. Open the referenced files alongside this document and follow
> along. The conceptual version is [../architecture/execution_flow.md](../architecture/execution_flow.md);
> this is the "read the actual code" version.

Files involved (in call order):

```text
pyproject.toml                       [project.scripts] entry
autoware_ml/cli/cli.py               Typer app + the `train` command
autoware_ml/utils/cli/helpers.py     run_lazy_script (lazy import + call)
autoware_ml/cli/runtime.py           run_hydra_entrypoint, prepare_runtime_environment
autoware_ml/scripts/train.py         the real @hydra.main entrypoint
autoware_ml/utils/runtime.py         instantiate_trainer / instantiate_callbacks / seed
```

---

## Step 0 — the console script

`pyproject.toml`:

```toml
[project.scripts]
autoware-ml = "autoware_ml.cli.cli:main"
```

`pip install -e .` creates an `autoware-ml` executable that calls
`autoware_ml.cli.cli:main`, which runs the Typer app (`app()`). Typer inspects `argv[1]`
(`train`) and dispatches to the matching command function.

Why Typer and not plain argparse: Typer gives shell tab-completion and typed options for
free, and the CLI is deliberately kept import-light so completion is fast (no torch/Hydra at
import time).

---

## Step 1 — the `train` command (`autoware_ml/cli/cli.py:205`)

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

Key observations:

- **`allow_extra_args` + `ignore_unknown_options`** is what lets you append raw Hydra
  overrides (`trainer.max_epochs=100`, `model.optimizer.lr=1e-4`) after the known flags.
  They arrive as `ctx.args` and are forwarded as `extra_args`.
- **`--weights` / `--resume-checkpoint` become Hydra overrides** (`+weights=[...]`,
  `+resume_checkpoint=...`). The `+` *adds* a new top-level config key. `scripts/train.py`
  later reads them with `cfg.get("weights")` / `cfg.get("resume_checkpoint")`.
- **The `+`** matters: it adds a key that isn't in the schema. Overriding an *existing* key
  uses no `+`. (See [config_flow.md](config_flow.md).)
- `deploy` and `test` have the same shape; only `stage=` and the entrypoint module differ
  (`deploy` also passes `checkpoints=weights` for multi-checkpoint MLflow lineage).

`run_lazy_script` (`autoware_ml/utils/cli/helpers.py`) is deliberately trivial:

```python
def run_lazy_script(module_path, function_name, *args, **kwargs):
    module = importlib.import_module(module_path)   # torch/Hydra imported HERE, not at CLI startup
    return getattr(module, function_name)(*args, **kwargs)
```

---

## Step 2 — the Hydra bridge (`autoware_ml/cli/runtime.py:277`)

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

This function does **two** things before calling the real entrypoint:

### 2a. Pre-create the MLflow run (`prepare_runtime_environment:188`)

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

**Why a throwaway compose?** To read `cfg.logger.tracking_uri` and create the MLflow run
*before* the real job, so the run id and directory are known up front. The real job then
reuses them via the two environment variables. For `deploy`/`test` this is also where the
**run lineage** (parent/source run) is resolved (`resolve_deploy_lineage`,
`resolve_lineage_context`).

### 2b. Pin the run directory

`AUTOWARE_ML_HYDRA_RUN_DIR` is consumed by `configs/defaults/modules/run.yaml`:

```yaml
hydra:
  run:
    dir: ${oc.env:AUTOWARE_ML_HYDRA_RUN_DIR,mlruns/${user_config_name:${hydra:job.config_name}}/_hydra/...}
```

So the Hydra output dir equals the MLflow run dir — checkpoints, config snapshots, and Hydra
logs all land in the same place. `scripts/train.py:73` asserts they match.

---

## Step 3 — the real entrypoint (`autoware_ml/scripts/train.py:56`)

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

**The single most important line to internalize:** every major object is
`hydra.utils.instantiate(cfg.<section>)`. `cfg.datamodule`, `cfg.model`, `cfg.logger`, and
(inside the helpers) `cfg.callbacks` and `cfg.trainer` are all just config trees; Hydra reads
each `_target_` and constructs the object. There is no registry lookup. See
[config_flow.md](config_flow.md) for how those trees are built.

`instantiate_trainer` (`utils/runtime.py`) is a thin wrapper:

```python
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=trainer_logger or False,
                                  default_root_dir=root_dir)   # cfg.trainer._target_ == lightning.Trainer
```

`instantiate_callbacks` iterates `cfg.callbacks.values()` and instantiates each, skipping
`LearningRateMonitor` when there's no logger and rewriting `ModelCheckpoint.dirpath` to the
MLflow checkpoint dir.

---

## Step 4 — `trainer.fit()` and beyond

From `trainer.fit(...)` you are in Lightning. Lightning calls the hooks your model inherited
from `BaseModel` (`models/base.py`):

```text
setup → configure_optimizers            (once)
per training batch:
    on_after_batch_transfer  →  training_step  →  loss.backward()  →  optimizer.step()
per validation epoch:
    validation_step ×N  →  on_validation_epoch_end (metric suites)  →  ModelCheckpoint(monitor=val/loss)
```

Those hooks are detailed in [important_classes.md](important_classes.md) (BaseModel) and
[../training/training_loop.md](../training/training_loop.md).

---

## The whole chain, condensed

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

## Common debugging cases

| Symptom | Where the break is | What to check |
| ------- | ------------------ | ------------- |
| Unknown option / arg rejected | The `train` command signature | It's a typed flag; raw Hydra overrides go *after* known flags as `ctx.args` |
| `+weights` "could not override" | You used `+` on an existing key, or omitted `+` on a new one | New keys need `+`; existing keys don't |
| "Hydra work directory does not match…" | `train.py:73` | `AUTOWARE_ML_HYDRA_RUN_DIR` vs the MLflow run dir; usually a stale env var |
| MLflow run created but training never starts | The throwaway compose in `prepare_runtime_environment` failed | Re-run with the same config + `--cfg job` to see composition errors |
| `optimized_metric ... was not logged` | `train.py:167` | The monitored metric key must actually be logged (e.g. `val/loss`) |
| Weird import error only under the CLI | A module imported at CLI top-level (breaks lazy design) | Keep heavy imports inside functions / the `scripts/` entrypoints |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Add a `--foo` flag to `train` | Add a typed `typer.Option` in `train()`, translate to a Hydra override, read via `cfg.get("foo")` in `scripts/train.py` |
| Run `test` right after `fit` | Add `trainer.test(...)` at the end of `scripts/train.py:main` |
| Change where runs/artifacts land | `configs/defaults/modules/run.yaml` + `utils/mlflow_helpers.py` |
| Add a new command (e.g. `benchmark`) | Add a `@app.command` in `cli.py` + a `scripts/benchmark.py` with a `main`, dispatch via `run_hydra_entrypoint` or `run_lazy_script` |

---

**Next:** [config_flow.md](config_flow.md) — how the `cfg` that drives all of this is composed.
