# Optimizer & Scheduler

> **本文涵蓋內容：** `configure_optimizers()` 如何將 config 中的 *partial* 轉變成一個
> 運作中的 optimizer + LR schedule、帶有各群組（per-group）覆寫的參數群組（parameter
> groups）、`total_steps` 的自動填入，以及 scheduler 目錄。
> 先備知識：[training_loop.md](training_loop.md)。

---

## 1. 為什麼 optimizer/scheduler 是「partial」

每一個子模組（`backbone`、`head`、…）都是在模型的 constructor 執行*之前*由 Hydra
建構的。但 optimizer 在那個時候還無法建構 — 它需要 `model.parameters()`，而這在模型
被建構出來之前是不存在的。所以 optimizer 和 scheduler 是用 `_partial_: true` 來設定的，
這會讓 Hydra 產生一個 `functools.partial`（一個*工廠函式*），而不是一個實例（instance）：

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

`BaseModel.__init__` 會把這些存成 `self.optimizer_partial` / `self.scheduler_partial`。
它們會在稍後、模型（以及其參數）已經存在之後，於 `configure_optimizers()` 內部被
*呼叫（called）*。

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

建構函式（`utils/optimizer.py:126`）：

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

`call_configured_factory` 在呼叫 PyTorch 之前，還會把 partial 裡綁定的 OmegaConf
容器（`DictConfig`/`ListConfig`）**具體化（materialize）**成純 Python 物件 — 否則
torch 會對 OmegaConf 的型別吃不消。回傳值可能是一個裸的（bare）optimizer，也可能是
Lightning 的 `{"optimizer", "lr_scheduler"}` dict。

---

## 3. 參數群組（每個模組各自的 LR / weight decay）

預設情況下，模型只會暴露一個群組（`BaseModel.build_optimizer_groups`，`base.py:175`）：

```python
def build_optimizer_groups(self):
    return {"default": [p for p in self.parameters() if p.requires_grad]}
```

`build_optimizer_param_groups`（`utils/optimizer.py:77`）會把具名（named）群組轉換成
PyTorch 的參數群組，並套用以群組名稱為 key 的**各群組覆寫（per-group overrides）**：

```python
# config:
model:
  optimizer_group_overrides:
    img_backbone: { lr: 0.00002 }   # e.g. StreamPETR gives the image backbone a smaller LR
```

如果一個模型想要更精細的控制，它可以覆寫 `build_optimizer_groups()`，回傳多個具名
群組（例如 StreamPETR 把 `img_backbone` 拆出來；PTv3 把 `block` 拆出來）。不明的
覆寫名稱會立刻丟出例外（`utils/optimizer.py:104`）— 打錯字會直接明顯地失敗。

---

## 4. `total_steps` 的自動填入

某些 scheduler（例如 `OneCycleLR`、`IterWarmupEpochCosineLR`）需要知道 optimizer
的總步數 — 這取決於 dataset 大小、batch size、epoch 數，以及累積（accumulation），
也就是執行期（runtime）才知道的資訊。建構函式會利用 Lightning 的
`trainer.estimated_stepping_batches` 計算出這個值，並**只有在** scheduler 的
signature 有宣告 `total_steps`、且該值尚未在 config 中被綁定時，才會把它注入：

```python
if "total_steps" in sig.parameters and "total_steps" not in bound:
    scheduler_kwargs["total_steps"] = estimated_stepping_batches
```

所以你永遠不需要自己手動計算步數 — 只要在你的 scheduler 的 signature 中宣告
`total_steps`，框架就會幫你填入。

---

## 5. `scheduler_config` — 依 step 還是依 epoch

Lightning 需要知道多久該讓 scheduler 走一步。這個中繼資料（metadata）是透過
`scheduler_config` 傳遞的，並會被合併進 `lr_scheduler` 這個 dict：

```yaml
model:
  scheduler_config:
    interval: step        # step every optimizer step (vs "epoch")
    # frequency: 1
    # monitor: val/loss   # for ReduceLROnPlateau-style schedulers
```

逐 iter（per-iter）的 scheduler（OneCycle、PTv3、FRNet、StreamPETR 對齊版本）會使用
`interval: step`；單純的逐 epoch scheduler 則會省略它。

---

## 6. Scheduler 目錄（`autoware_ml/utils/schedulers/`）

| Scheduler | 形狀 | 主要參數 | 被誰使用 |
| --------- | ----- | -------- | ------- |
| `CyclicCosineAnnealingLR` | cosine warmup → cosine decay | `warmup_epochs`, `decay_epochs`, `max_lr_factor`, `min_lr_factor` | CenterPoint, StreamPETR |
| `IterWarmupEpochCosineLR` | per-iter linear warmup × per-epoch cosine | `total_steps` (auto), `max_epochs`, `warmup_iters` | some det3d |
| `LinearWarmupCosineAnnealingLR` | linear warmup → cosine | `warmup_epochs`, `max_epochs`, `warmup_start_lr`, `eta_min` | — |
| `CosineAnnealingLR` / `CyclicMomentumScheduler` | standard | — | — |
| `torch.optim.lr_scheduler.OneCycleLR` (stock) | one-cycle | `max_lr` (per group) | seg PTv3 |

完整呈現 `CyclicCosineAnnealingLR`（`utils/schedulers/cyclic_cosine_annealing.py`）—
這是一個很適合拿來當作範本、寫自己的 scheduler 時可以參考的模型：

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

注意 LR 是以**optimizer 基礎 `lr` 的倍率（factor）**來表示的（例如 `max_lr_factor: 10.0`
→ 峰值 = `10 × lr`），所以你可以獨立於基礎 LR 之外，單獨調整這個 schedule 的形狀。

Optimizer 本身就是標準 torch 的東西（`AdamW` 等），搭配 `_partial_: true`。

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| `Optimizer must be provided` | config 中沒有 `optimizer`，或缺少 `_partial_` | 加入 optimizer 區塊，並帶上 `_partial_: true` |
| Optimizer 在錯誤的時機被建構 / 拿不到參數 | 忘了加 `_partial_: true`（Hydra 立刻呼叫了它） | 加上 `_partial_: true` |
| `Unknown optimizer group override(s)` | 覆寫名稱 ≠ `build_optimizer_groups` 回傳的群組名稱 | 讓名稱一致，或覆寫 `build_optimizer_groups` |
| Scheduler 對 `total_steps` 報錯 | scheduler 需要它，但 trainer 尚未接上 | 該值會從 `trainer.estimated_stepping_batches` 填入；確認你是在 `fit` 之中 |
| LR 沒有逐 step 改變 | 缺少 `scheduler_config.interval: step` | 對逐 iter 的 scheduler 加上這個設定 |
| optimizer 內出現 OmegaConf 型別錯誤 | 容器未被具體化（materialize） | 使用框架提供的建構函式（`call_configured_factory` 會處理這件事） |
| LR 與預期相比過高/過低 | 這些倍率會乘上基礎 `lr` | 記住峰值 = `lr × max_lr_factor` |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 更改基礎 LR / weight decay | `model.optimizer.lr=...`, `model.optimizer.weight_decay=...` |
| 讓不同模組使用不同 LR | 覆寫 `build_optimizer_groups` + 設定 `optimizer_group_overrides` |
| 切換 scheduler | 更改 `model.scheduler._target_` 及其參數（保留 `_partial_: true`） |
| 逐 step 還是逐 epoch 更新 | 設定 `model.scheduler_config.interval` |
| 全新的 schedule | 在 `utils/schedulers/` 中新增一個 `LRScheduler` 子類別；如果需要的話，在 `__init__` 中宣告 `total_steps` |

---

**Next:** [loss_design.md](loss_design.md) — loss 存放的位置，以及它們是如何被計算出來的。
