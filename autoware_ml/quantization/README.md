# Quantization 模組:宣告式量化(PTQ / QAT,INT8 / FP8)

> 讀者設定:第一次接觸這個框架的人。讀完你應該能:對現有模型跑一次 PTQ、看懂
> placement 輸出、知道 QAT 什麼時候值得、以及替新模型接上量化。
> 部署面(export / TensorRT / verification)見
> [`../deployment/README.md`](../deployment/README.md)。

## 0. 三分鐘版

```bash
# 1) 先看不燒 GPU 的 placement(強烈建議)
autoware-ml quantize --config-name experiments/.../<model>_int8 \
  --weights <fp_training.ckpt> +quantization.dry_run=true

# 2) PTQ:替換模組 → 校準 → 存自描述 checkpoint
autoware-ml quantize --config-name experiments/.../<model>_int8 --weights <fp_training.ckpt>

# 3) 部署(不需要任何 quantization config——ckpt 自己知道自己是什麼)
autoware-ml deploy --config-name experiments/.../<model>_int8 --weights <.../ptq.ckpt>
```

心智模型一句話:**模型宣告「哪裡可以量化」(架構事實,寫在 code),config 只做減法
(skip / disable),engine 負責執行並把每筆決策記錄在 checkpoint 裡。**

## 1. 核心概念:決策與執行分離

### 1.1 `QuantRules`:模型的量化宣告(`plan.py`)

每個支援量化的模型有一個 `main_modules/<model>/quantization.py`(現例:PTv3 57 行、
BEVFusion 67 行),核心是:

```python
MODEL_QUANT_RULES = QuantRules(
    quantize_submodules={
        "backbone": ("conv",),                      # 全走 config 的 default_precision
        "seg3d_head": {"conv": None, "linear": "fp8"},  # per-kind 釘死精度
    },
    recipes=(...),  # 架構 recipe(預設全部;class 比對,不中則不動作)
)
```

- key 是模型**頂層屬性名**;模型沒有該屬性 → 靜默跳過(一份 rules 服務多個變體)。
- 「哪些 kind 可換」是架構事實,屬於 code;config 的 `skip_quantize` /
  `disable_recipes` **只能減不能加**。

### 1.2 Plan → PlacementRecord:每筆決策可審查

`build_quantization_plan` 把 rules + config 展開成逐模組的 `PlacementDecision`
(哪個模組、換成什麼、**為什麼**),全部進 `PlacementRecord`。`dry_run` 印的就是它;
quantize 完它內嵌進 checkpoint——這就是「自描述」:deploy/test 讀 record 重建結構,
**不讀 `cfg.quantization`**(same-plan-everywhere 不變量)。

### 1.3 執行端:modelopt registry(`core/`)

模組替換走 modelopt 的 quantized-module registry(`core/modelopt.py`),不自己維護
替換表;校準(`core/calibration.py`)由 config 的 `quantization.calibration` 區塊驅動
(方法/樣本數)。state_dict 的 quantizer key 是 modelopt 慣例(`*input_quantizer.*`)。

### 1.4 Recipes:matcher + action(`recipes/`)

架構特例(如「某類 block 的第二個 conv 跳過」)寫成 recipe:**class 比對**決定在哪
生效、`RECIPE_ATTACHERS` registry 註冊 action;plan import 時驗證 recipe 名,忘記註冊
直接爆。沒有比中任何模組的 recipe 是 no-op(不會誤傷)。

## 2. PTQ workflow(標準路)

1. `dry_run` 確認 placement(§0)。
2. `quantize`:載 FP training ckpt(**未 fuse** 的;BN fusion 由框架在量化前做,見
   `core/fusion.py`)→ 替換 → 校準 → 存 `ptq.ckpt`。
3. `deploy` 該 ckpt;verification 的 INT8 容差哲學見 deployment README §4。

## 3. QAT workflow(需要時才用)

```yaml
quantization:
  mode: qat
  qat:
    freeze_unquantized: true   # 預設;不凍會崩(見下)
    schedule: cosine           # 或 one_cycle / constant;peak lr 建議 1e-5
```

付過學費的三件事:

1. **`freeze_unquantized: true` 是預設且必要**:未量化層吸收梯度漂移 → 輸出越過
   凍結的 amax → clip 歸零梯度 → 正回饋崩潰(CenterPoint 實測 mAP 0.81 → 0.007)。
2. **lr 要小**:freeze + peak 1e-5 整個 epoch 穩定;1e-4 會在 epoch 後段非線性惡化
   (300 步探針看不出來,recipe 驗證必須整 epoch)。
3. **QAT 不保證贏 PTQ**:CenterPoint 上兩者持平(0.8128 vs 0.8132),正式路徑是
   PTQ;PTv3(attention 模型)QAT 有感(+0.7 mIoU vs PTQ)。先 PTQ,證明不夠再 QAT。

## 4. INT8 vs FP8 怎麼選(兩模型交叉驗證)

| 層類 | 建議 | 證據 |
| --- | --- | --- |
| conv(CNN backbone) | INT8 | CenterPoint/BEVFusion:mAP 損失 <0.02、backbone 3.9→3.5 ms |
| linear(attention/FFN) | **FP8,不要 INT8** | PTv3:INT8 linear 賠 6.4 mIoU,FP8 只賠 0.37;BEVFusion FFN FP8 ±0 |
| attention 內部(QK/AV matmul) | 不量 | 斷 TRT fused-MHA,反而 +0.3 ms |

FP8 走 modelopt 的 trt-domain 自訂 op、per-tensor scale、max 校準;ONNX Runtime
載不了 FP8 圖(該 experiment 關 onnx backend)。完整數據:
`work_dirs/reviews/fp8-quantization-README.md`。

## 5. 新增量化模型 checklist

1. **宣告**:寫 `main_modules/<model>/quantization.py`(§1.1),接上模型的
   `build_quantization_plan`。
2. **config**:`_int8` / `_fp8` experiment(照抄現例改名),含 `skip_quantize` 與
   verification scenarios。
3. **dry_run** 看 placement,再 quantize,再 deploy。
4. export log 有 `Quantized chain breaks` 警告 → 處理方式見 deployment README §3 末。

**五個已知陷阱(都付過學費,附實例):**

1. **attention 投影在校準期抓不到**:訓練態 `nn.MultiheadAttention` 的 qkv 是 packed
   Parameter,export 態 `q/k/v/out_proj` Linear 在校準之後才誕生;`out_proj` 是
   forward 被 fast path 繞過的特殊 Linear(walker 已在框架層拒換)。要量它們 =
   校準前先換成 export 態 attention(未實作)。→ 詳:
   `models/detection3d/main_modules/bevfusion/quantization.py` docstring。
2. **輸入端層對 INT8 敏感**:吃 raw / scatter 特徵的第一段(CenterPoint stage 0)
   量了掉 ~1.2 mAP,release recipe 一直 skip。新模型對輸入段做 leave-one-out。
   → 實例:centerpoint `_int8.yaml` 的 `skip_quantize` 註解。
3. **linear 用 FP8 不用 INT8**(§4)。
4. **ORT 跑不了 plugin stage 與 FP8 op**:前者 `torch_fallback_backends`,後者關
   onnx backend。→ 實例:ptv3/base.yaml、bevfusion `_fp8` config。
5. **verification tolerance 實測校準**:量化 stage 的 raw-logit 跨 backend 差是預期,
   metric 相等才是 gate;fail 訊息會給建議值。→ 實例:centerpoint `_int8.yaml`。

**一條不可動的地基**:ONNX 圖上 Q/DQ 保持 **fp32-typed(island)**。fp16-typed Q/DQ
(opset 19 合法)踩 TRT 10.8/10.16 缺陷:fp16 合併 scale subnormal → 融合 kernel 產
NaN、build 零警告。island 規則與證據:deployment README §3、
`work_dirs/reviews/fp16-typed-qdq-nogo.md`(金絲雀 = PTv3 INT8 QAT)。

## 6. 檔案地圖

```text
quantization/
  plan.py            QuantRules / PlacementDecision / build plan(宣告與展開)
  config.py          quantization config schema(mode/calibration/ptq/qat)
  checkpoint.py      自描述 ckpt 的存讀(placement_record 內嵌)
  loader.py          按 record 重建量化模型(deploy/test 入口)
  qat_callback.py    QAT:epoch-0 校準、freeze_unquantized、schedule
  core/
    modelopt.py      modelopt registry 對接(模組替換)
    calibration.py   校準執行
    fusion.py        量化前 BN fusion
    replace.py       walker(含 out_proj 等拒換保護)
    descriptors.py / quantizer_state.py   模組描述與 quantizer 狀態
  recipes/
    attach.py        RECIPE_ATTACHERS registry
    quant_blocks.py  recipe 實作(class-matched)
```

## 7. 歷史與深挖

架構是 2026-08~09 從「四種變異機制、三層隱性決策」重構而來;重構診斷、命名決議
(rename 對照表)、21 個 review QA 都在 git history 與
`work_dirs/reviews/`(`migration-framework-review-README.md`、
`ptq-qat-verification-README.md`、`fp8-quantization-README.md`、
`quantization-vs-modelopt-comparison-README.md`、`fp16-typed-qdq-nogo.md`)。想知道「為什麼長這樣」
先查這些,再挖 git log。
