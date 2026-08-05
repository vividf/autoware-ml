# Autoware-ML Onboarding Guide

> A learning-oriented walkthrough of the Autoware-ML framework for new contributors.
> Goal: in 1–2 weeks you should be able to **read most of the code, add a new model,
> and debug a training or deployment issue** on your own.

This guide is **not** a replacement for the reference docs under `docs/framework/`,
`docs/user-guide/`, and `docs/models/`. Those tell you *what the API is*. This guide
tells you *why the framework is shaped the way it is* and *how the pieces fit together*,
so the reference docs start making sense.

Everything here is grounded in the source. When a claim points at code it uses a
`path:line` reference you can open directly. **The source is the source of truth** — if
this guide and the code ever disagree, the code wins; please fix the guide.

---

## Who this is for

An engineer who:

- knows **PyTorch** (tensors, `nn.Module`, autograd, a training loop),
- has some **autonomous-driving / 3D perception** background (point clouds, bounding
  boxes, cameras),
- but is **new to this framework** — and possibly new to PyTorch Lightning and Hydra.

If you came from the older `tier4/AWML` (MMDetection3D-based) repo, read
[architecture/framework_overview.md](architecture/framework_overview.md) first — the
mental model is different and reusing your AWML instincts will actively mislead you.

---

## The 5-level mental model

You do not need to understand everything at once. Learn it in layers. Each level is a
prerequisite for the next.

| Level | You understand… | Read |
| ----- | --------------- | ---- |
| **1. Framework structure** | What lives where and why; the stack (Lightning + Hydra + MLflow) | [architecture/framework_overview.md](architecture/framework_overview.md) |
| **2. Data flow** | How one sample becomes a GPU-ready batch | [architecture/data_flow.md](architecture/data_flow.md), then [dataset/](dataset/dataset_pipeline.md) |
| **3. Training pipeline** | How `trainer.fit()` turns a batch into a weight update | [architecture/execution_flow.md](architecture/execution_flow.md), then [training/](training/training_loop.md) |
| **4. Model integration** | How a model plugs into the `BaseModel` contract | [model/](model/model_architecture.md) |
| **5. Deployment optimization** | How a checkpoint becomes a TensorRT engine | [deployment/](deployment/export_pipeline.md) |

---

## Suggested reading order

1. [architecture/framework_overview.md](architecture/framework_overview.md) — the world-view and the repository map. **Start here.**
2. [architecture/data_flow.md](architecture/data_flow.md) — follow one sample end to end.
3. [architecture/execution_flow.md](architecture/execution_flow.md) — what happens when you run `autoware-ml train`.
4. [code_walkthrough/entry_point.md](code_walkthrough/entry_point.md) — the literal code trace, function by function.
5. [code_walkthrough/config_flow.md](code_walkthrough/config_flow.md) — how a Hydra config composes.
6. [code_walkthrough/important_classes.md](code_walkthrough/important_classes.md) — the classes you will touch most, as a reference card.
7. Then go deep per area: [dataset/](dataset/dataset_pipeline.md) → [model/](model/model_architecture.md) → [training/](training/training_loop.md) → [evaluation/](evaluation/evaluation_pipeline.md) → [deployment/](deployment/export_pipeline.md).

---

## Directory of this guide

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

## The one-paragraph summary (read this every time you get lost)

Autoware-ML is a **PyTorch Lightning + Hydra** framework. A **Hydra YAML config**
(`autoware_ml/configs/tasks/<task>/<model>/<variant>_<dataset>.yaml`) fully describes a
run. The CLI (`autoware-ml train|test|deploy`) composes that config and calls
`hydra.utils.instantiate()` on each section to build a **`DataModule`**, a **model**
(subclass of `BaseModel`, which *is* a `LightningModule`), Lightning **callbacks**, an
**MLflow logger**, and a **`Trainer`** — then calls `trainer.fit()`. The model implements
only two abstract methods, `forward()` and `compute_metrics()` (which must return a
`"loss"`); everything else — the training/val/test/predict steps, optimizer setup, metric
logging, and ONNX export — is inherited from `BaseModel`. **Almost every object is built
from a config `_target_`, so most bugs are configuration bugs, and the fix is usually in
YAML, not Python.**
