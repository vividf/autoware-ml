# Optimizer & Scheduler

> **What this covers:** how `configure_optimizers()` turns config *partials* into a running
> optimizer + LR schedule, parameter groups with per-group overrides, the `total_steps`
> auto-fill, and the scheduler catalog.
> Prerequisite: [training_loop.md](training_loop.md).

---

## 1. Why optimizer/scheduler are "partials"

Every sub-module (`backbone`, `head`, …) is built by Hydra *before* the model constructor
runs. But the optimizer can't be built then — it needs `model.parameters()`, which don't exist
until the model is constructed. So the optimizer and scheduler are configured with
`_partial_: true`, which makes Hydra produce a `functools.partial` (a *factory*) instead of an
instance:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true            # → functools.partial(AdamW, lr=1e-4, weight_decay=0.01)
  lr: 0.0001
  weight_decay: 0.01
scheduler:
  _target_: autoware_ml.utils.schedulers.cyclic_cosine_annealing.CyclicCosineAnnealingLR
  _partial_: true            # → partial(CyclicCosineAnnealingLR, warmup_epochs=8, ...)
  warmup_epochs: 8
  decay_epochs: 22
  max_lr_factor: 10.0
  min_lr_factor: 0.0001
```

`BaseModel.__init__` stores these as `self.optimizer_partial` / `self.scheduler_partial`. They
are *called* later, inside `configure_optimizers()`, once the model (and thus its parameters)
exists.

---

## 2. `configure_optimizers` → `build_lightning_optimizer_config`

```python
# BaseModel.configure_optimizers (models/base.py:395)
def configure_optimizers(self):
    if self.optimizer_partial is None:
        raise ValueError("Optimizer must be provided.")
    return build_lightning_optimizer_config(
        self, self.optimizer_partial, self.scheduler_partial,
        optimizer_group_overrides=self.optimizer_group_overrides,
        scheduler_config=self.scheduler_config,
        estimated_stepping_batches=self.trainer.estimated_stepping_batches if self._trainer is not None else None,
    )
```

The builder (`utils/optimizer.py:126`):

```python
def build_lightning_optimizer_config(model, optimizer_factory, scheduler_factory=None, *,
                                     optimizer_group_overrides=None, scheduler_config=None,
                                     estimated_stepping_batches=None):
    param_groups = build_optimizer_param_groups(model, optimizer_group_overrides)   # §3
    optimizer = call_configured_factory(optimizer_factory, params=param_groups)     # partial(...)(params=...)

    if scheduler_factory is None:
        return optimizer

    scheduler_kwargs = {"optimizer": optimizer}
    sig = inspect.signature(scheduler_factory)
    bound = _get_partial_keywords(scheduler_factory)
    if estimated_stepping_batches is not None and "total_steps" in sig.parameters and "total_steps" not in bound:
        scheduler_kwargs["total_steps"] = estimated_stepping_batches               # §4 auto-fill
    scheduler = call_configured_factory(scheduler_factory, **scheduler_kwargs)

    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **materialize_partial_kwargs(dict(scheduler_config or {}))}}
```

`call_configured_factory` also **materializes OmegaConf containers** (`DictConfig`/`ListConfig`
bound in the partial) into plain Python before calling PyTorch — otherwise torch chokes on
OmegaConf types. Returns either a bare optimizer or Lightning's `{"optimizer", "lr_scheduler"}`
dict.

---

## 3. Parameter groups (per-module LR / weight decay)

By default the model exposes one group (`BaseModel.build_optimizer_groups`, `base.py:175`):

```python
def build_optimizer_groups(self):
    return {"default": [p for p in self.parameters() if p.requires_grad]}
```

`build_optimizer_param_groups` (`utils/optimizer.py:77`) turns named groups into PyTorch param
groups and applies **per-group overrides** keyed by group name:

```python
# config:
model:
  optimizer_group_overrides:
    img_backbone: { lr: 0.00002 }   # e.g. StreamPETR gives the image backbone a smaller LR
```

If a model wants finer control, it overrides `build_optimizer_groups()` to return multiple
named groups (e.g. StreamPETR splits out `img_backbone`; PTv3 splits out `block`). Unknown
override names raise immediately (`utils/optimizer.py:104`) — a typo fails loudly.

---

## 4. The `total_steps` auto-fill

Some schedulers (e.g. `OneCycleLR`, `IterWarmupEpochCosineLR`) need the total number of
optimizer steps — which depends on dataset size, batch size, epochs, and accumulation, i.e.
runtime info. The builder computes it from Lightning's `trainer.estimated_stepping_batches`
and injects it **only if** the scheduler's signature declares `total_steps` and it wasn't
already bound in config:

```python
if "total_steps" in sig.parameters and "total_steps" not in bound:
    scheduler_kwargs["total_steps"] = estimated_stepping_batches
```

So you never hand-compute step counts — declare `total_steps` in your scheduler's signature
and the framework fills it.

---

## 5. `scheduler_config` — step vs epoch

Lightning needs to know how often to step the scheduler. That metadata is passed via
`scheduler_config` and merged into the `lr_scheduler` dict:

```yaml
model:
  scheduler_config:
    interval: step        # step every optimizer step (vs "epoch")
    # frequency: 1
    # monitor: val/loss   # for ReduceLROnPlateau-style schedulers
```

Per-iter schedulers (OneCycle, PTv3, FRNet, StreamPETR parity) use `interval: step`; simple
epoch schedulers omit it.

---

## 6. The scheduler catalog (`autoware_ml/utils/schedulers/`)

| Scheduler | Shape | Key args | Used by |
| --------- | ----- | -------- | ------- |
| `CyclicCosineAnnealingLR` | cosine warmup → cosine decay | `warmup_epochs`, `decay_epochs`, `max_lr_factor`, `min_lr_factor` | CenterPoint, StreamPETR |
| `IterWarmupEpochCosineLR` | per-iter linear warmup × per-epoch cosine | `total_steps` (auto), `max_epochs`, `warmup_iters` | some det3d |
| `LinearWarmupCosineAnnealingLR` | linear warmup → cosine | `warmup_epochs`, `max_epochs`, `warmup_start_lr`, `eta_min` | — |
| `CosineAnnealingLR` / `CyclicMomentumScheduler` | standard | — | — |
| `torch.optim.lr_scheduler.OneCycleLR` (stock) | one-cycle | `max_lr` (per group) | seg PTv3 |

`CyclicCosineAnnealingLR` (`utils/schedulers/cyclic_cosine_annealing.py`) in full — a good
model for writing your own:

```python
class CyclicCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs=8, decay_epochs=12, max_lr_factor=10.0,
                 min_lr_factor=1e-4, last_epoch=-1):
        self.warmup_epochs, self.decay_epochs = warmup_epochs, decay_epochs
        self.max_lr_factor, self.min_lr_factor = max_lr_factor, min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:            # phase 1: base_lr → base_lr*max_lr_factor
            start_factor, end_factor = 1.0, self.max_lr_factor
            t_cur, t_max = self.last_epoch, self.warmup_epochs
        else:                                                # phase 2: peak → base_lr*min_lr_factor
            start_factor, end_factor = self.max_lr_factor, self.min_lr_factor
            t_cur, t_max = self.last_epoch - self.warmup_epochs, self.decay_epochs
        lr_factor = end_factor + 0.5 * (start_factor - end_factor) * (1 + math.cos(math.pi * t_cur / t_max))
        return [base_lr * lr_factor for base_lr in self.base_lrs]
```

Note LRs are expressed as **factors of the optimizer's base `lr`** (e.g. `max_lr_factor: 10.0`
→ peak = `10 × lr`), so you tune the schedule shape independently of the base LR.

Optimizers themselves are stock torch (`AdamW`, etc.) with `_partial_: true`.

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `Optimizer must be provided` | no `optimizer` in config, or missing `_partial_` | add optimizer block with `_partial_: true` |
| Optimizer built at wrong time / gets no params | forgot `_partial_: true` (Hydra called it immediately) | add `_partial_: true` |
| `Unknown optimizer group override(s)` | override name ≠ a group from `build_optimizer_groups` | match names, or override `build_optimizer_groups` |
| Scheduler errors about `total_steps` | scheduler needs it but trainer not attached | it's filled from `trainer.estimated_stepping_batches`; ensure you're inside `fit` |
| LR not changing per step | missing `scheduler_config.interval: step` | add it for per-iter schedulers |
| OmegaConf type error inside optimizer | container not materialized | use the provided builder (`call_configured_factory` handles it) |
| LR too high/low vs intent | factors multiply the base `lr` | remember peak = `lr × max_lr_factor` |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Change base LR / weight decay | `model.optimizer.lr=...`, `model.optimizer.weight_decay=...` |
| Different LR per module | override `build_optimizer_groups` + set `optimizer_group_overrides` |
| Switch scheduler | change `model.scheduler._target_` + its args (keep `_partial_: true`) |
| Per-step vs per-epoch stepping | set `model.scheduler_config.interval` |
| A brand-new schedule | add an `LRScheduler` subclass in `utils/schedulers/`; declare `total_steps` in `__init__` if you need it |

---

**Next:** [loss_design.md](loss_design.md) — where losses live and how they're computed.
