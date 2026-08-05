# ONNX 與 TensorRT

> **本文涵蓋內容：** 匯出的內部機制 — `torch.onnx.export`（dynamo 與 legacy 兩種模式）、動態
> 形狀（dynamic shapes）、ONNX 圖修改器（graph modifier）、TensorRT engine 的建構，以及需要
> 對應 TensorRT 外掛（plugin）的自訂 op。這是最深入、最貼近車輛端（vehicle-facing）的一層。
> 先備知識：[export_pipeline.md](export_pipeline.md)。

這裡提到的所有程式碼都位於 `autoware_ml/utils/deploy.py`（匯出 + TRT）、
`autoware_ml/utils/onnx_modifiers.py`（圖的修改）以及 `autoware_ml/ops/`（自訂 op）之中。

---

## 1. ONNX 匯出（`export_to_onnx`，`utils/deploy.py:328`）

```python
def export_to_onnx(model, input_sample, onnx_cfg, input_param_names,
                   output_names_override, dynamic_axes_override, output_path):
    dynamo = onnx_cfg.get("dynamo", True)
    dynamic_shapes = build_dynamic_shapes(onnx_cfg, input_param_names) if dynamo else None
    dynamic_shapes = normalize_dynamic_shapes_for_model(model, dynamic_shapes) if dynamo else None
    dynamic_axes = (dynamic_axes_override or build_dynamic_axes(onnx_cfg)) if not dynamo else None
    input_names  = list(onnx_cfg.get("input_names", input_param_names))
    output_names = list(output_names_override or onnx_cfg.get("output_names", ["output"]))

    register_scatter_reduce_onnx_symbolic(opset_version=int(onnx_cfg.opset_version))   # :360

    export_kwargs = {"model": model, "args": input_sample, "f": str(output_path),
                     "input_names": input_names, "output_names": output_names,
                     "opset_version": onnx_cfg.opset_version, "dynamo": dynamo,
                     "do_constant_folding": onnx_cfg.get("do_constant_folding", True)}
    export_kwargs["dynamic_shapes" if dynamo else "dynamic_axes"] = dynamic_shapes if dynamo else dynamic_axes
    torch.onnx.export(**export_kwargs)                                                  # :377
    # if a `.onnx.data` shard was written, merge it back into a single file
```

這其實只是一次 `torch.onnx.export` 呼叫，但透過 `deploy.onnx.dynamo` 可以選擇**兩種模式**：

| 模式 | `dynamo` | 動態維度的實現方式 | 使用時機 |
| ---- | -------- | ---------------- | ---- |
| **Dynamo**（預設） | `true` | `dynamic_shapes`（`torch.export.Dim`） | 現代基於 `torch.export` 的匯出方式 |
| **Legacy** | `false` | `dynamic_axes`（name→dim 對應表） | 依賴舊版 ONNX symbolic function 的模型（例如 CenterPoint、TransFusion、FRNet） |

在匯出之前，它會註冊一個共用的 symbolic（`register_scatter_reduce_onnx_symbolic`），讓
`aten::scatter_reduce` 對應到標準的 ONNX `ScatterElements`。匯出之後，如果 `torch.onnx` 寫出了
外部資料分片（external-data shard，`.onnx.data`），`merge_onnx_external_data` 會將它們合併回
單一個自我完備（self-contained）的 `.onnx` 檔案。

### 動態形狀（Dynamic shapes）（dynamo 模式，`build_dynamic_shapes:218`）

Config 會將輸入名稱對應到 {維度索引 → symbolic}。有兩種寫法：

```yaml
deploy:
  onnx:
    dynamic_shapes:
      input_tensor: { 2: height, 3: width }        # shorthand: dim 2 = "height", dim 3 = "width"
      points:
        0: { name: num_points, min: 2 }             # explicit, with bounds
```

每一項都會變成一個 `torch.export.Dim(name, min=?, max=?)`。對於 `forward(*args)` 這類
wrapper，`normalize_dynamic_shapes_for_model` 會將結構再往內包一層，因為 `torch.export` 要求
`dynamic_shapes` 必須與位置參數（positional-arg）的 pytree 結構一致。未知的參數名稱會導致錯誤。

### 動態軸（Dynamic axes）（legacy 模式，`build_dynamic_axes:285`）

使用相同的 config 形式，但產生的是 legacy 版本的 `{tensor_name: {dim: name}}` 對應表。
CenterPoint 的 config：

```yaml
deploy:
  onnx:
    dynamo: false
    opset_version: 17
    modules:
      pts_voxel_encoder_centerpoint:
        input_names: [input_features]
        output_names: [pillar_features]
        dynamic_axes:
          input_features: { 0: num_voxels, 1: num_max_points }
```

---

## 2. 選用的 ONNX 圖修改器（`utils/onnx_modifiers.py`）

有些 TensorRT 的限制，透過改寫 ONNX 圖來解決，會比修改模型本身來得容易。匯出之後，如果設定了
`deploy.onnx.modify_graph`，`modify_onnx_graph` 就會（透過 Hydra 的 `_target_`）實例化該修改器
並套用：

| 修改器 | 改寫的內容 | 原因 |
| -------- | ---------------- | --- |
| `TopKConstantKModifier` | 將 TopK 節點的 `K` 輸入 → 改為編譯期常數（compile-time constant） | TensorRT 不接受由 argsort 衍生出的動態 `K` |
| `AttentionScaleToDivModifier` | `Mul(q, scale)` → `Div(q, 1/scale)` | 讓 attention 的縮放（scaling）對 TRT 更友善 |
| `TransHeadTensorRTModifier` | 結合上述兩者 | TransFusion 風格的 decoder head（用於 PTv3 detection） |

Config 範例（PTv3 detection 的 `det3d_head` 模組）：

```yaml
deploy:
  onnx:
    modules:
      det3d_head:
        modify_graph:
          _target_: autoware_ml.utils.onnx_modifiers.TransHeadTensorRTModifier
          k: ${num_proposals}
          topk_node_name_substring: /bbox_head/TopK
          attention_node_name_substring: /bbox_head/decoder
```

這些修改器直接操作原始的 `onnx` protobuf + `numpy_helper`（不依賴
`onnx_graphsurgeon`/`onnxscript`）。首選的做法仍然是把匯出邏輯保留在模型內
（`build_export_specs`）；圖修改器則是專門用來應付 TRT 特有怪癖（quirk）的逃生出口
（escape hatch）。

---

## 3. TensorRT engine 建構（`build_tensorrt_engine`，`utils/deploy.py:484`）

```python
def build_tensorrt_engine(onnx_path, deploy_cfg, output_path):
    tensorrt_cfg = deploy_cfg.tensorrt
    builder, network, parser, config = create_tensorrt_builder_config(tensorrt_cfg)   # :492
    parse_onnx_file(parser, onnx_path)                                                 # :493
    profile = create_optimization_profile(builder, tensorrt_cfg)                       # :495
    if profile is not None:
        config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)              # :500
    output_path.write_bytes(serialized_engine)
```

Builder 的設定（`create_tensorrt_builder_config:427`）：

```python
import tensorrt as trt
trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, "")                        # loads TRT plugins, incl. custom autoware:: ops
builder = trt.Builder(trt_logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))   # :434
parser  = trt.OnnxParser(network, trt_logger)
config  = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, tensorrt_cfg.get("workspace_size", 1 << 30))
```

有兩個設計選擇值得留意：

- **`init_libnvinfer_plugins`** 會載入 TensorRT 的外掛註冊表（plugin registry） — 這正是車輛端
  自訂的 `autoware::` 運算子外掛（對應 §5 中的 op）能夠被 parser 使用的原因。
- **`STRONGLY_TYPED` network** — 精度（precision）是取自 **ONNX tensor 的 dtype**，而非透過
  builder 旗標（flag）設定。詳見 §4。

### 優化設定檔（Optimization profile）（`create_optimization_profile:457`）

對於動態輸入，你需要提供 TensorRT min/opt/max 形狀，讓它能夠預先規劃（pre-plan）kernel：

```yaml
deploy:
  tensorrt:
    workspace_size: 8589934592     # 8 GiB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # TRT optimizes most aggressively for opt_shape
        max_shape: [1, 3, 512, 512]
```

每個輸入都必須同時提供這三者（`min`/`opt`/`max`），否則會拋出錯誤。請將 `opt_shape` 設為你在
車輛上實際使用的典型解析度（resolution）。

---

## 4. 精度（Precision）

因為這個 network 是**`STRONGLY_TYPED`**，FP16/FP32 是直接內建（baked in）於 ONNX 圖的 tensor
dtype 之中 — 這個 builder 中**沒有**任何 `config.set_flag(FP16/INT8)` 呼叫，也沒有 INT8
校準器（calibrator）。因此：

- 若要匯出半精度（half-precision）的 engine，ONNX 本身就必須帶有 fp16 的 tensor（透過 half /
  autocast 匯出產生），而不是靠 builder 旗標。
- `trainer.precision`（例如 `bf16-mixed`）是**訓練（training）**精度，在這裡完全沒有作用。
- 這個 TRT builder 中沒有 INT8 PTQ 路徑。

---

## 5. 自訂 op 與 ONNX ↔ TensorRT 之間的橋接（`autoware_ml/ops/`）

有些運算（operation）沒有標準的 ONNX/TensorRT 對應版本，因此這個 repo 提供了自訂 op。每一個
自訂 op 都定義了一個 `torch.autograd.Function`，其 `forward` 會執行 eager 模式的 kernel，而其
`symbolic` 則會輸出一個**標準的** ONNX op，或是一個帶有 **`autoware::` 命名空間**的自訂 op，
由車輛端對應的 TensorRT 外掛來實作。

| 套件 | 原生（Native）？ | 匯出橋接方式 |
| ------- | ------- | ------------- |
| `ops/bev_pool/` | **是 — CUDA**（`bev_pool_ext`，由 `src/*.cu,*.cpp` 建構而成） | `QuickCumsumCuda.symbolic` → `g.op("autoware::QuickCumsumCuda", ...)`（BEVFusion 相機→BEV pooling） |
| `ops/indexing/` | 否（包裝 `torch.unique`/`torch.sort`） | `_Unique.symbolic` → `autoware::CustomUnique`；`_Argsort.symbolic` → `autoware::Argsort`（僅在 `torch.onnx.is_in_onnx_export()` 為真時） |
| `ops/segment/` | 否 | `_SegmentCSR.symbolic` → `autoware::SegmentCSR`；以及 `scatter_reduce` → 標準 ONNX `ScatterElements`（於 `export_to_onnx` 中註冊） |
| `ops/spconv/` | 否（依賴外部套件 `spconv`） | autograd `Function` 輸出 `autoware::` sparse-conv op |
| `ops/voxelization/` | 沒有自訂 ONNX op | 純 PyTorch 實作的 voxelization（用於前處理，不會被匯出為自訂 op） |

唯一經過編譯的原生 op 是 `bev_pool`（`ops/build.py` → `bev_pool_ext`，架構 gencode 為
sm_80/86/89/90/120）。其餘都是帶有 ONNX symbolic 的 Python wrapper。

**讓這一切能夠端到端（end to end）運作的鏈路是：** 模型使用該 op → 該 op 的 `symbolic` 將一個
`autoware::` 節點寫入 ONNX 圖中 → 在建構 engine 時，`init_libnvinfer_plugins` 載入對應的外掛 →
車輛端的 TensorRT runtime 執行它。這也是為何像 PTv3（sparse conv）這樣的模型，在目標 runtime
缺少對應外掛時，會被標記為僅支援 ONNX — 詳見 [export_pipeline.md](export_pipeline.md)。

---

## 6. 端到端流程：checkpoint → ONNX → engine

```text
scripts/deploy.py
  load ckpt → model.eval()
  resolve_export_specs → {module: ExportSpec(module, args, names, ...)}
  per module:
    merge_module_onnx_cfg          shared deploy.onnx.* + deploy.onnx.modules.<name>.*
    export_to_onnx                 register scatter symbolic → torch.onnx.export(dynamo|legacy) → {module}.onnx
    (modify_onnx_graph)            optional TRT-oriented graph rewrite
    build_tensorrt_engine          init plugins → STRONGLY_TYPED network → parse ONNX
                                   → optimization profile (min/opt/max) → build_serialized_network → {module}.engine
```

---

## 常見除錯情境

| 症狀 | 原因 | 修正方式 |
| ------- | ----- | --- |
| ONNX 匯出在自訂 op 上失敗 | 該 op 缺少 symbolic，或匯出時沒有 `is_in_onnx_export` 保護（guard） | 使用 `ops/` 中已具備 symbolic 的 op，或自行新增一個 |
| `torch.export` 的 dynamic_shapes 結構錯誤 | wrapper 的 `forward(*args)` 需要更深一層的巢狀結構 | 由 `normalize_dynamic_shapes_for_model` 處理；請檢查 wrapper 的簽章（signature） |
| 未知的 dynamic-shape 參數 | 名稱不在匯出輸入名稱之中 | 與 `input_param_names` 保持一致 |
| TRT 在 TopK / argsort 上解析（parse）錯誤 | 動態 `K`、不支援的模式 | 透過 `modify_graph` 加入 `TopKConstantKModifier` / `TransHeadTensorRTModifier` |
| TRT 出現 "no plugin for autoware::X" | 外掛未載入 / 未針對目標建構 | 確認 `init_libnvinfer_plugins` 已執行，且車輛端 runtime 有內建該外掛；否則就只匯出 ONNX |
| Engine 比預期更大/更慢 | workspace 太小、`opt_shape` 設定錯誤 | 提高 `workspace_size`；將 `opt_shape` 設為實際的解析度 |
| fp16 沒有生效 | STRONGLY_TYPED 是從 ONNX 讀取 dtype | 以 fp16 匯出 ONNX；不要依賴 builder 旗標 |
| `CUDA not available` | TRT 需要 GPU | 在具備 CUDA 的硬體上執行 |

---

## 常見修改情境

| 我想要… | 這麼做 |
| ---------- | ------- |
| 切換 dynamo ↔ legacy 匯出模式 | `deploy.onnx.dynamo=true|false` |
| 新增/調整動態維度 | `deploy.onnx.dynamic_shapes`（dynamo 模式）或 `dynamic_axes`（legacy 模式） |
| 變更 opset | `deploy.onnx.opset_version` |
| 新增針對 TRT 的圖修正 | 將 `deploy.onnx.modules.<m>.modify_graph._target_` 設為某個修改器 |
| 調整 engine 設定檔（profile） | `deploy.tensorrt.input_shapes.<in>.{min,opt,max}_shape`、`workspace_size` |
| 新增自訂 op | 在 `ops/` 底下新增一個 `torch.autograd.Function`，並附上輸出 `autoware::<Op>` 的 `symbolic`（以及對應的 TRT 外掛） |

---

**onboarding 指南到此結束。** 你現在已經完整走過整個框架，從 CLI 一路追蹤到 TensorRT engine。
回到 [README](../README.md) 查看閱讀地圖（reading map），並謹記這條指導原則：**原始碼才是真正
的真實來源（source of truth） — 請對照程式碼進行驗證，並在文件與程式碼產生落差時修正這些文件。**
