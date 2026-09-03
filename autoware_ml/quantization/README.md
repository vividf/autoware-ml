# Quantization 模組:架構重構(Plan + PlacementRecord)

> 本文件是**設計決策與重構歷史的記錄**;想理解現行架構與使用方式,
> 請讀 `docs/framework/quantization.md`。
>
> 狀態:**已實作**(2026-08)。§2 保留重構前的診斷作為背景;§3–§5 描述的方案即現行架構,
> 檔案對應見 §6。正式使用文件見 `docs/framework/quantization.md`。

---

## 1. 現狀地圖

```text
autoware_ml/quantization/
├── config.py            # Hydra `quantization` 區塊的 typed view(唯一一次 parse;recipe 名稱驗證)
├── checkpoint.py        # 自描述 checkpoint:config + placement record 內嵌在 state_dict 旁(無 sidecar)
├── loader.py            # 由 checkpoint 內嵌描述重建量化樹 → 比對 placement record → 載入
├── qat_callback.py      # QAT:Lightning callback(plan prepare、epoch-0 校準、frozen-amax、on_save 內嵌描述)
├── plan.py              # QuantRules / QuantizationPlan / PlacementRecord(stage 間的唯一介面)
├── core/
│   ├── replace.py       # Conv/ConvTranspose/Linear → Quant* 子類替換引擎;expand_skip_quantize
│   ├── modules/         # QuantConv2d / QuantConvTranspose2d / QuantLinear
│   ├── descriptors.py   # per-precision descriptor 表(唯一寫 bit width 的地方)
│   ├── calibration.py   # Calibrator(collect_stats / amax / .calib cache)
│   ├── fusion.py        # dense Conv+BN fusion(BN → Identity)
│   ├── utils.py         # disable/validate/count quantizers、ONNX export 設定
│   └── backend.py       # modelopt backend 介面
├── recipes/
│   ├── attach.py        # residual-add / eSE / maxpool 的 quantizer 附掛 + hook 安裝
│   └── quant_forwards.py # BasicBlock/SparseBasicBlock/ConvNeXt/OSA/eSE 的 forward 替換物件
└── sparse/fusion.py     # SparseConv+BN fold(FP16 sparse encoder deploy)

模型端宣告:
models/detection3d/main_modules/centerpoint/quantization.py
  → CENTERPOINT_QUANT_RULES + build_centerpoint_quantization_plan()

呼叫點(三處都必須建出同一棵樹):
  scripts/quantize.py             (PTQ / QAT 產出)
  quantization/qat_callback.py    (QAT)
  quantization/loader.py          (build_model 偵測到量化 checkpoint 時載入;deploy/test 皆走此)
```

## 2. 診斷:亂的根源

亂的感受不是「replace vs recipe」本身,而是以下三件事疊加:

### 2.1 四種結構變異機制,各自一套語彙和怪癖

| 機制 | 位置 | 對 module tree 的效果 |
| --- | --- | --- |
| 子類替換 | `core/replace.py` | `Conv2d` → `QuantConv2d`(整個 module 換掉) |
| forward monkeypatch | `recipes/attach.py` + `quant_forwards.py` | module 不換,`forward` 換成 hook 物件,quantizer 以 submodule/attribute 附掛 |
| 模組包裝 | `QuantBeforePool` | `MaxPool2d` → `QuantBeforePool(quantizer, pool)` |
| 權重手術 | `core/fusion.py` / `sparse/fusion.py` | Conv 權重改寫、BN → Identity |

子類替換內部還有**兩種 clone 方式**:Conv 走乾淨的 `__init__` + weight copy
(`_rebuild_*_as_quant`),Linear 走 `vars()` transplant(`clone_as_quant_by_transplant`,
`__new__` + 屬性搬運)。code 內已留 TODO 指向統一。

讀者每碰到一種機制就要重新理解一次「它怎麼改樹、state_dict 會長什麼樣」。

### 2.2 「哪裡量化、怎麼量化」的決策散在三層,而且是隱性的

- **Config 層**:`skip_quantize`(glob、子樹語意)、`disable_recipes`。
- **模型層**(`centerpoint_quantization.py`):硬編哪些 tower 換 Conv、哪些換 Linear
  (`pts_backbone` → Conv+Linear、`pts_neck`/`bbox_head` → Conv、`pts_voxel_encoder` → Linear)。
- **引擎層**(`recipes/attach.py`):`attach_residual_add_recipe` 用**類別名稱字串 / substring 比對**
  (`_RESIDUAL_BLOCK_CLASSES`)掃全模型,命中後用四層 if/elif 啟發式決定 residual
  quantizer 是新建還是共用(`downsample`? `conv1._input_quantizer`? `depthwise_conv`?
  `concat[0]`?)。

「依同模型有不同效果」正是這裡的產物:最終的 Q/DQ 配置是 **emergent** 的,
沒有任何一個地方能回答「這個模型最後被放了哪些 Q/DQ、各是為什麼」——只能跑一次、讀 log。

### 2.3 最重要的不變量靠紀律維護,不靠機器檢查

「PTQ producer、QAT callback、deploy loader 三處必須建出**一模一樣的樹**」
(state_dict 才對得起來)目前靠:

- 三個呼叫點都記得呼叫同一個 `build_quantization_plan(config).prepare(model)`;
- 記得「`expand_skip_quantize` 要在 `prepare` **之後**」這類順序注釋(prepare 會改樹,
  glob 在前後解析結果不同);
- skip_quantize 語意被實作兩次:替換引擎的 skip test + 事後 `disable_quantizers_in`。

`QuantizationScheme`/`QuantizationPlan` 抽象方向正確,但它只把「呼叫同一段 code」
變成契約,**沒有產出可以比對的證據**。

## 3. 提案:Plan-as-manifest —— 把「決策」和「執行」拆開

核心想法一句話:現在的 `prepare(model)` 是**邊決定邊動手**;
改成先產出一份完整的置換清單(manifest),再由一小組統一的 primitive 執行。

```text
QuantRules(每個模型一份宣告)+ QuantizationConfig
        │  resolve(model)          ← 只讀模型、不改模型
        ▼
QuantPlan = [PlacementDecision, ...]   ← 可 print、可存檔、可 diff
        │  apply(model)             ← 只執行、不做決策
        ▼
    mutated model
```

### 3.1 `PlacementDecision`:每筆決策寫清楚「哪個模組、哪種 transform、為什麼」

```text
("pts_backbone.blocks.0.conv1", ReplaceModule(QuantConv2d),
     reason="tower rule: pts_backbone conv+linear")
("pts_neck.deblocks.1",         KeepFP16,
     reason="skip_quantize pattern 'pts_neck.deblocks.*'")
("backbone.stage2.osa1",        PatchBlockForward(QuantOSAModuleForward,
                                  quantizers=[fresh, share:concat[0].input_q]),
     reason="recipe 'residual_add': matched _OSA_module")
```

直接換來三個能力:

1. **`--dry-run`**:量化前印出全模型 INT8/FP16/recipe 配置表。
   「不同模型有不同效果」從黑箱變成一頁報表。
2. **Manifest 內嵌在 checkpoint 裡**(`checkpoint["quantization"]`,連同 config):loader
   重建樹後 resolve 出自己的 manifest,**逐筆比對、不合就 hard-fail**。
   「同一棵樹」的不變量從紀律變成機器檢查;deploy/test 不再需要 `quantization` config。
3. **測試對 manifest 斷言**,不必跑完整量化再翻 module tree。

### 3.2 執行端收斂成三個 primitive,各只有一種寫法

| Primitive | 吸收現有的 | 備註 |
| --- | --- | --- |
| `ReplaceModule` | 原 `quant_conv_module` / `quant_linear_module`(已刪除,收斂為 `replace_quantizable_modules`) | Linear 的 `vars()` transplant clone 保留(切換前需 mAP parity 驗證,見 §5) |
| `WrapModule` | `QuantBeforePool` | |
| `PatchBlockForward` | residual/eSE/OSA hooks | monkeypatch **保留**(見 §5 取捨),但升格為有名字、有契約的 primitive |

BN fusion(dense 與 sparse)同樣以 decision 形式進 plan;
`QuantizationScheme` 這一層可整個被 plan 吸收。

### 3.3 Recipe 拆成 matcher + action,消滅 substring 比對與內嵌啟發式

現在 `attach_residual_add_recipe` 一個函式混著「找誰」(類名 substring)和「做什麼」
(四路 quantizer 共用啟發式)。改成明確的註冊表:

```text
BasicBlock     → ResidualAddRecipe(share=conv1.input_q, fresh_if=has_downsample)
ConvNeXtBlock  → ResidualAddRecipe(share=depthwise_conv.input_q)
_OSA_module    → OSAConcatRecipe(branch_inputs=..., identity_reuse=concat[0])
eSEModule      → SingleQeSERecipe(pool_input + mul_gate)
```

每個 block 類型一行宣告;共用/新建 quantizer 的規則變成 recipe 參數。
新架構要支援 = 加一行註冊,而不是往 150 行的函式裡再塞一個 elif。

### 3.4 模型端從「命令式膠水」變成「宣告」

`centerpoint_quantization.py` 的本體縮成:

```python
CENTERPOINT_RULES = QuantRules(
    towers={
        "pts_backbone":      (Conv, Linear),
        "pts_neck":          (Conv,),
        "bbox_head":         (Conv,),
        "pts_voxel_encoder": (Linear,),
    },
    recipes=("residual_add", "ese", "maxpool"),
)
```

Generic resolver 負責 rules + config(`skip_quantize` / `disable_recipes`)→ plan:

- `skip_quantize` 的 match**只發生在一處**(`match_skip_quantize_roots`),plan 直接記
  `skip_quantize` decision;
- 「先插 quantizer 再 `disable_quantizers_in` 關掉」的流程保留——這是**刻意的**:
  state_dict key 佈局不能變,既有 checkpoint 才能繼續載。disable pass 不改
  state_dict key,因此留在 manifest(樹結構記錄)範圍之外;
- 新模型要支援量化 = 一份 `QuantRules` 宣告,不是一個新的膠水檔。

### 3.5 統一詞彙

現在 recipes(=hooks)、schemes(=策略物件)、plan(=scheme 清單)三個詞互相打架。
建議固定為:

> **transform**(primitive)→ **recipe**(matcher + action)→
> **rules**(每模型宣告)→ **plan**(resolve 出的 manifest)

## 4. 遷移路徑與實作狀態

| Phase | 內容 | 狀態 |
| --- | --- | --- |
| **1** | manifest 記錄 + `quantization.dry_run` + producer/loader manifest 比對 | ✅ 已實作 |
| **2** | `attach_residual_add_recipe` 拆成 `ResidualBlockSpec` 註冊表,殺掉 substring 比對的 monolith | ✅ 已實作 |
| **3** | skip_quantize 單點解析(`match_skip_quantize_roots` + `expand_skip_quantize`);clone 機制統一**保留 Linear transplant**(切換需 mAP parity 驗證,見 §5) | ✅/保留 |
| **4** | 模型端改宣告式 `QuantRules`;`schemes/` 套件併入 `plan.py` | ✅ 已實作 |

### 驗收標準(每次後續改動同樣適用)

> 2026-08-31 rename(§8)改了 payload key 佈局;rename 前的量化 ckpt 皆為實驗性,
> 直接重新 quantize。以下標準適用於 rename 之後的改動。

1. `state_dict` key set **逐字不變**(rename 後的 PTQ/QAT checkpoint 必須能繼續載);
2. weight-amax 黃金值不變(CenterPoint PTQ tutorial 的驗證流程);
3. mAP parity(PyTorch / ONNX / TensorRT 三 backend);
4. ONNX graph 的 Q/DQ 節點集合不變(export 走 hook 的 trace 行為)。

## 5. 取捨聲明(誠實版)

- **forward monkeypatch 不會消失。** 在不引入 FX / `torch.export` graph rewrite 的前提下,
  residual-add 的 Q placement 沒有更乾淨的做法;而 spconv + 動態 shape 使 graph trace
  在這個 codebase 成本與風險不成比例。本重構把它從「散落的技巧」升格為
  「一個有名字、有契約的 primitive」(`patch_forward`),而不是假裝能消滅它。
- **Linear 的 `vars()` transplant clone 保留。** 換成 rebuild 路徑需要 mAP parity
  驗證(需要 GPU + 資料的容器環境),在驗證前不切換;`clone_as_quant_by_transplant`
  的 docstring 記錄了這個 TODO。
- **「插了再 disable」的 skip_quantize recipe quantizer 保留。** 改成「不插」會改變
  state_dict key 佈局,破壞既有 checkpoint 相容性;post-load 的 disable pass 不改
  state_dict key,所以刻意留在 manifest 範圍之外。
- 本重構收斂的是:**四種機制 → 有名字的 transform 詞彙表;三層隱性決策 → 一份
  看得見的清單;紀律維護的不變量 → 機器比對的 manifest。**

## 6. 實作對應(檔案地圖)

| 概念(§3) | 實作 |
| --- | --- |
| `PlacementDecision` / `PlacementRecord` | `quantization/plan.py` |
| `QuantRules`(宣告)+ `QuantizationPlan`(resolve+apply+record) | `quantization/plan.py` |
| CenterPoint 宣告 | `models/.../centerpoint/quantization.py` 的 `CENTERPOINT_QUANT_RULES`(原 `quant_model` + `CenterPointDenseScheme` 膠水已刪除) |
| ReplaceModule primitive | `core/replace.py` 的 `replace_quantizable_modules`(kind table 驅動;舊的 `quant_conv_module`/`quant_linear_module` 已刪除) |
| recipe 註冊表(matcher+action) | `recipes/attach.py` 的 `_RESIDUAL_SPECS`(`ResidualBlockSpec`:quant_forward、share_from、fresh_if_downsample 都是宣告參數) |
| skip_quantize 單點解析 | `core/replace.py` 的 `match_skip_quantize_roots`(match)+ `expand_skip_quantize`(子樹展開) |
| placement record 內嵌 + 驗證 | `checkpoint.py`:quantize stage 把 `QuantizationDescription(config, placement_record)` 寫進 checkpoint(QAT 由 `QATCallback.on_save_checkpoint`);loader 以 `PlacementRecord.verify_matches` 硬性比對。`.calib` / `.manifest.json` sidecar 已取消 |
| dry-run | `quantization.dry_run=true`(CPU 即可,印全模型 placement 表後結束) |
| `schemes/` 套件 | 已刪除,seam 併入 `plan.py`(呼叫介面 `build_quantization_plan(config).prepare(model)` 不變) |
| 測試 | `tests/quantization/test_plan_record.py`(placement record/rules)+ `test_quantized_checkpoint.py`(內嵌描述 round-trip / 偵測 / drift)+ `test_tree_parity.py` 雙 prepare placement 對等性 |

## 7. 命名決定(2026-08 rename,無相容層)

判準:不與 PyTorch/Lightning 既有詞彙相撞;同字同義;config key 說語意不說實作。

| 舊名 | 新名 | 理由 |
| --- | --- | --- |
| `*ForwardHook`(五個 class)、`forward_hooks.py` | `Quant*Forward`、`quant_forwards.py` | 它們不是 torch hook(`register_forward_hook`),而是 forward 的量化版重寫 |
| `_install_forward_hook` | `_replace_block_forward` | 同上——動作是「取代」不是「掛 hook」 |
| `prepare_quantized_model`(loader) | `load_quantized_model` | 與 `QuantizationPlan.prepare` 撞字不同義;本質是 load |
| `transfer_to_quantization` | `clone_as_quant_by_transplant` | 點名機制(`vars()` transplant),與 `_rebuild_*_as_quant` 成對 |
| `disable_quantization`(小寫 class) | `set_quantizers_enabled()` + `quantizers_disabled()` ctx | PEP8;一物二用拆成兩個明確函式 |
| `attach_quant_add` / `attach_ese_quantizers` / `attach_maxpool_input_quantizer` | `attach_residual_add_recipe` / `attach_ese_recipe` / `attach_maxpool_recipe` | 統一 `attach_<recipe>_recipe`;"add" 是 ONNX 行話 |
| `CalibrationManager` | `Calibrator` | "Manager" 是空詞 |
| 文件用語 "producer" | "quantize stage" | 借用的 MQ 行話,repo 本有 quantize/deploy 的自然詞彙 |
| config `keep_fp16` | `skip_quantize` | 舊名說謊:被排除模組的實際精度由 deploy 端決定(現為 `deploy.onnx.precision`),不必然 fp16 |
| recipe 名 `add` | `residual_add` | config 裡單獨的 "add" 不可讀 |
| config `fused_checkpoint` | `weights_bn_fused` | 舊名讀起來像輸出;實義是「輸入權重已 BN-fused」 |
| config `calib_cache_path` / `calib_cache` | (已移除) | amax 已在 state_dict;checkpoint 自描述後不再需要 `.calib` cache |

`QATCallback`(Lightning 慣例)與 `QuantizationPlan.prepare`(三個 stage 共用的動詞)刻意不改。

## 9. 架構整理(2026-08-31,architecture review Phase 1–5)

依 `work_dirs/reviews/architecture-review-2026-08-31.md` 的決議實作,8-sample PTQ
bit-exact 驗證(state_dict keys、64 個 amax、weights 全零差)通過:

| 改動 | 內容 |
| --- | --- |
| precision 配線 | `config.default_precision` → plan → replace/attach → `core/descriptors.py` 的 per-precision 表;framework 中只有 descriptors 寫 bit width |
| descriptor class-attr 退役 | `default_quant_desc_*` class attributes 與 `ensure_quant_descriptors_initialized` 全刪;desc 一律由呼叫端顯式傳入 `init_quantizer` |
| Calibrator 收緊 | `forward_fn` 必填、校準 batch 失敗即 raise(不再靜默 skip)、`set_quantizer_fast`→`enable_torch_histogram` |
| optional-dep 機制刪除 | modelopt 已是必要依賴:`core/availability.py`、`QUANT_BACKEND_AVAILABLE`、`*_or_none` accessor(→`tensor_quantizer_cls()`)、export-trace bypass guard 全刪 |
| export/TRT primitive 歸位 | ONNX primitive → `deployment/onnx_export.py`、TRT builder → `deployment/backends/tensorrt_builder.py`(typed 簽名);`utils/deploy.py` 只剩 legacy ExportSpec(隨 Q5 刪檔) |
| MLflow 樣板 | deploy/quantize 兩個 main() 共用 `mlflow_run_scope` context manager |

## 8. 命名決定(2026-08-31 rename,無相容層)

由 21 個 `Question(vividf)` code review 疑問逐項決議(問答與選項記錄:
`work_dirs/reviews/naming-refactor-QA.md`)。rename 前產出的量化 checkpoint 皆為實驗性,
不做相容:以 `autoware-ml quantize` 重新產出即可。

| 舊名 / 舊行為 | 新名 / 新行為 | 理由 |
| --- | --- | --- |
| `QuantizationManifest`(`.record()`、`plan.manifest`) | `PlacementRecord`(`.add()`、`plan.placement_record`) | manifest 雖是 CS 標準語,團隊讀感不佳;record 直白 |
| checkpoint payload key `"manifest"` | `"placement_record"` | 同上;rename 前的 ckpt 皆為實驗性,直接重跑 quantize(不設相容層;內嵌格式亦無 version 欄位——config 的 unknown-key 檢查與 missing-key 已足以擋格式漂移) |
| `QuantRules.towers` | `QuantRules.quantize_submodules` | tower 是自造詞;實義是「頂層 submodule → 替換 kinds」 |
| `VALID_TOWER_KINDS` | `VALID_MODULE_KINDS` | 同上 |
| manifest reason `"tower rule: ..."` | `"submodule rule: ..."` | 同上 |
| `build_quantized_tree` | `build_quantized_model` | 它回傳的就是 model |
| `load_checkpoints_into_tree` | `_load_checkpoints_into_model`(私有) | 唯一呼叫者剩 loader 自己 |
| config `weights_bn_fused` + 雙路徑 | 移除:quantize 只吃未 fuse 的 FP training ckpt | AWML 轉換(BN-fused)ckpt 的量化路徑確定不再需要 |
| config `default_precision: str` | `Precision` enum(YAML 仍寫 `int8`) | 型別化;仍只支援 int8 |
| modelopt optional(`[quant]` extra、lazy import) | 必要依賴、import 全部上移 | 所有環境統一裝 modelopt(需重產 pixi.lock) |
| recipe 三連 `if` | `RECIPE_ATTACHERS` registry(attach.py)+ plan import 時比對 `VALID_RECIPES` | 加 recipe 只改一處,忘記註冊會在 import 時爆 |
| QAT 撿 `best.ckpt` hardcode 路徑 | `trainer.checkpoint_callback.best_model_path` | 檔名跟 callback config 走;deploy `--weights` 本就由使用者任選 |
| `_dry_run_manifest` | `_log_placement_dry_run` | 動詞說明它做的事(印表) |
| docstring 用語 seam / "the transform vocabulary the manifest speaks" | 白話("the single interface between ..."、"Transforms a placement record can contain");glue(glue code 常語)保留 | 可讀性 |

## 10. 新增量化模型 checklist(2026-09-04)

新 model 要走通 quantize → deploy(INT8/FP8),需要的全部工作與已知陷阱。
precision pass(island cast / AutoCast 路由)完全自動,model 端**零 precision 程式碼**。

### 步驟

1. **宣告量化面**:在 `main_modules/<model>/quantization.py` 寫 `QuantRules`
   (參考:PTv3 57 行、BEVFusion 67 行)。`quantize_submodules` 的 key 是 model
   的**頂層屬性名**;kind 用 tuple(全走 config 的 `default_precision`)或 mapping
   釘死 per-kind precision(如 BEVFusion 的 `{"conv": None, "linear": "fp8"}`——
   linear 永不 INT8,PTv3 實測 INT8 linear 賠 6 mIoU 換不到 latency)。
2. **寫 `_int8` / `_fp8` experiment config**(參考 centerpoint/ptv3/bevfusion 的現例):
   `quantization:` 區塊 + `skip_quantize` + verification scenarios。
3. **先 dry-run 再燒 GPU**:`quantize ... +quantization.dry_run=true` 印出完整
   placement record,確認替換的模組正是你要的。
4. `quantize` 產 ptq.ckpt(自描述)→ `deploy --weights <ptq.ckpt>` 評估。
   deploy/test 不讀 `cfg.quantization`。
5. export log 裡如果出現 **"Quantized chain breaks at ..."** 警告:那個 op 若量化
   可交換 → 加進 `_QDQ_COMMUTING_OPS`(連同 float-slot 表一行,import 檢查會強制)
   並重跑三模型 battery;若不可交換 → 加進 `_KNOWN_NON_COMMUTING_OPS` 消音。

### 五個已知陷阱(都付過學費,附實例)

1. **attention 的投影在校準期抓不到**:訓練態 `nn.MultiheadAttention` 的 qkv 是
   packed Parameter(不是 module),export 態 `q/k/v/out_proj` Linear 在
   `prepare_for_export` 才誕生(校準之後);`out_proj` 更是 forward 被 fast path
   繞過的 `NonDynamicallyQuantizableLinear`(walker 已在框架層拒換)。要量 attention
   投影 = 校準前先換 export 態 attention(未實作的 attention-recipe 前置)。
   → 實例與完整說明:`models/detection3d/main_modules/bevfusion/quantization.py` docstring。
2. **輸入端層對 INT8 敏感,照 release recipe skip**:CenterPoint backbone stage 0
   量了掉 ~1.2 mAP(輸入 pseudo-image 動態範圍大),AWML release recipe 一直是
   skip 的——遷移時漏過一次。加新 model 時對「吃 raw/scatter 特徵的第一段」做
   leave-one-out 檢查。→ 實例:`configs/experiments/detection3d/centerpoint/
   voxel024_..._int8.yaml` 的 skip_quantize 註解。
3. **linear 量化選 FP8 不選 INT8**:兩模型交叉驗證(PTv3 −0.37 vs INT8 −6.4 mIoU;
   BEVFusion FFN ±0)。FP8 走 trt-domain 自訂 op、per-tensor scale、max 校準,
   framework 已全通。→ `work_dirs/reviews/fp8-quantization-README.md`。
4. **ONNX Runtime backend 跑不了 plugin stage 與 FP8 op**:含 plugin 的 stage 宣告
   `torch_fallback_backends`,FP8 experiment 直接關 onnx backend(ORT 連圖都載不了)。
   → 實例:ptv3/base.yaml(onnx disabled 註解)、bevfusion `_fp8` config。
5. **verification tolerance 是實測校準的,不是猜的**:量化/FP16 stage 的 raw-logit
   跨 backend 差是預期行為,mAP/mIoU 相等才是真 gate。首跑 fail 時,錯誤訊息會
   給建議 gate 值(observed×1.25);把 observed 記進 config 註解。
   → 實例:centerpoint `_int8.yaml` scenarios 註解。

### 一條不可動的地基

**Q/DQ 保持 fp32-typed(island)是刻意且承重的設計**:fp16-typed Q/DQ(opset 19 合法、
ORT 算得對)會踩 TRT 10.8/10.16 的缺陷——fp16 合併 scale 落入 subnormal 時融合 kernel
產生 NaN、build 零警告。完整證據與重測工具:`work_dirs/reviews/fp16-typed-qdq-nogo.md`
(金絲雀 = PTv3 INT8 QAT)。island 的運作規則(誰進島、cast 放哪、每條規則的實測代價)
見 `deployment/onnx/precision.py` 的 docstrings。
