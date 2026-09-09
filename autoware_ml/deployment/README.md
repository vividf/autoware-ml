# Deployment 模組:從 checkpoint 到 TensorRT engine

> 讀者設定:第一次接觸這個框架的人。讀完你應該能:跑一次 deploy、看懂產物、
> 知道出錯時去哪裡找原因、以及替新模型接上這條 pipeline。
> 量化(PTQ/QAT)另有專文:[`../quantization/README.md`](../quantization/README.md);
> 訓練與資料面見 `docs/contributing/adding-models.md`。

## 0. 三分鐘版

這個模組做一件事:**把訓練好的 PyTorch 模型變成 TensorRT engine,並「證明」它沒有變壞**。
一條命令走完全部:

```bash
autoware-ml deploy \
  --config-name experiments/detection3d/centerpoint/<experiment> \
  --weights <checkpoint.ckpt>
```

它依序做五件事,每一步的產物都落在 experiment 目錄:

```text
export      每個 GraphStage 一份 <stage>.onnx
precision   fp16 化(自動選路:AutoCast / Q/DQ island cast / 原樣)
build       每份 onnx 一顆 <stage>.engine(TensorRT,一律 strongly typed)
verify      跨 backend 逐 tensor 比對(pytorch vs onnx vs tensorrt),過不了就 FAIL
evaluate    三個 backend 各跑一次完整 metric(mAP/mIoU)+ 每 stage latency 表
```

心智模型一句話:**模型自己宣告「我怎麼拆成可匯出的圖」(stage graph),框架負責
把每張圖推過 export→build→verify→evaluate,三種 backend 用同一條 pipeline 執行。**

## 1. 核心概念

### 1.1 Stage graph:模型自述怎麼拆(`stages.py`)

一個模型的 deploy 面 = 一個 `build_stages()` 方法,回傳 stage 序列。只有兩種 stage:

- **`GraphStage`**:一張可匯出的子圖 = 一份 ONNX = 一顆 engine。宣告
  `name / module / inputs / outputs`,inputs/outputs 的名字**就是** ONNX 的 IO 名,
  值從 `StageContext`(一個跨 stage 的 name→tensor 字典)取放。
- **`TorchStage`**:不可匯出的膠水(前處理、voxelize、scatter……),永遠跑 PyTorch,
  簽名 `fn(context) -> {name: value}`。

為什麼要拆:因為真實模型不是一張圖——中間有 sparse conv(需要 plugin)、有動態
shape 的索引計算、有根本不該進圖的預處理。stage graph 把「哪裡可以是圖、哪裡必須是
torch」變成模型的**宣告**,pipeline 照宣告執行,誰都不用改框架。

`GraphStage` 的進階欄位(用到才看):`torch_fallback_backends`(某 backend 跑不了這
張圖時退回 torch module,例:spconv 圖在 ONNX Runtime)、`onnx_dynamic_axes`(點雲類
模型天生的動態維度)、`onnx_transforms`(這張圖固有的匯出後重寫,如 bias+activation
摺進 plugin 節點)、`output_fields`(最終 stage 的輸出如何餵給 `assemble_predictions`)。

### 1.2 Backend 抽象:同一條 pipeline,三種執行體(`pipeline.py`, `backends/`)

`StagedPipeline` 對每個 backend 用同一套 stage 序列跑推論,差別只在 GraphStage 的
執行體是誰:

| backend | GraphStage 跑什麼 | 用途 |
| --- | --- | --- |
| `pytorch` | 原 torch module | 基準真值 |
| `onnx` | ONNX Runtime session | 驗證匯出圖的語意 |
| `tensorrt` | TRT engine | 交付形態 |

artifact 命名規則:`artifact_path(output_dir, stage_name, backend)` →
`<experiment>/<stage>.onnx` / `<stage>.engine`。latency 表裡每個 stage 一行、
`model_graphs` 一行(所有 GraphStage 合計)。

### 1.3 「same plan everywhere」不變量

量化模型的 checkpoint 是**自描述**的(placement record 內嵌),所以 `deploy` 和
`test` **不讀 `cfg.quantization`**——給什麼 ckpt 就 deploy 什麼。這保證訓練、量化、
部署三處看到的是同一個模型結構,歷史上的「校準時圖長 A 樣、匯出時長 B 樣」類 bug
被這個不變量整類消滅。

### 1.4 CLI:一個命令名、每個 config family 一個實作

`deploy` / `test` / `train` / `quantize` 由 config 路徑前綴(`experiments/...`)分派
到對應 family 的實作。所以不管什麼模型,命令長得一樣。

## 2. 一次 deploy 實際發生什麼

1. **build_stages()**:載入 ckpt(量化 ckpt 會先按 placement record 重建量化結構),
   模型回傳 stage 序列;`validate_stages` 檢查名字唯一、宣告完整。
2. **export**:每個 GraphStage `torch.onnx.export`(opset 由 `deploy.onnx.opset_version`
   決定——框架預設 21,現行三個 experiment 都 pin 17),IO 名即宣告名。
3. **precision pass**(`onnx/precision.py`,自動路由,模型端零程式碼):

   | 圖的事實 | 走哪條 | 原因 |
   | --- | --- | --- |
   | 有自訂 domain(plugin) | 自家 island cast(island-aware:圖裡若也有 Q/DQ,島照樣成立) | AutoCast 用 TRT parser 型別推導,不認 plugin op |
   | 有 Q/DQ(INT8/FP8) | 自家 island cast(fp32 島 + fp16 海) | AutoCast 拒收 Q/DQ 模型;island 是正確性地基,見 §3 |
   | 純圖 | modelopt AutoCast | 有數值守門(逐節點比對容差) |
   | `deploy.onnx.precision: fp32` | 原樣 | |

   判定順序:**先看 custom domain,再看 Q/DQ**——plugin 圖無論有沒有 Q/DQ 都走同一條
   island cast,兩者同時成立時不會走 AutoCast。同一步驟裡,`deploy.onnx.modify_graph`
   在 precision 之前跑(modifier 是照 fp32 匯出圖寫的),stage 自己宣告的
   `onnx_transforms` 在之後跑(`keep_topk_in_fp16` 改的正是 precision pass 插入的 cast:
   讓 TopK 直讀 fp16 heatmap,再把選出的 k 個 values cast 回 fp32 給原消費者 / graph output,
   所以消費者契約與 artifact ABI 不變,少掉的只有整張 heatmap 的那顆 cast)。

4. **TensorRT build**(`backends/tensorrt_builder.py`):**一律 strongly typed**——
   engine 的精度由 ONNX 圖的型別決定,不由 builder flag 猜。這是刻意決策:weak-typed
   加 `FP16` flag 會讓 TRT 的 kernel 自選精度,量化模型上曾實測翻車;strongly typed
   把精度變成**圖上可審查的事實**。plugin(`libautoware_tensorrt_plugins.so`)在
   build 前載入。
5. **verification**(§4)→ 6. **evaluation**(§5)。

## 3. Precision:fp16 的海、fp32 的線性島

量化圖的 fp16 化是**全圖轉型 + 一條例外**:conv 家族的 Q/DQ 連 scale 一起轉 fp16(opset 19
起合法),只有 **餵 Gemm/MatMul 的 Q/DQ** 保持 fp32-typed(「線性島」)。

**誰進島**(`_quantized_island_names`):每個 DQ 若其輸出(直接或穿過 Transpose/Reshape 等
純 layout op)到達 Gemm/MatMul,則 {該 DQ、它的 Q、兩者的 scale/zero-point producer、
layout hop、線性 op 本體} 進島,整段零 cast。其餘 Q/DQ 是海。

**為什麼按 op 類型切**(`_LINEAR_OPS` 註解、`work_dirs/reviews/uniform-fp16-exception-rule.md`):

| 事實 | 量測 |
| --- | --- |
| fp16-typed INT8 **Gemm** 在 TRT 10.8/10.16 會產 NaN | 合併 scale `s_x·s_w[c]` 掉到 fp16 subnormal 時融合 kernel 出事;PTv3 head mIoU 0.734→0.075,build 零警告 |
| conv kernel 免疫 | CenterPoint/BEVFusion 26 顆 conv 模組 combined scale 同樣 subnormal,uniform fp16 的 mAP 與 island 版相同 |
| 不能按 node 名單 | 「4 顆兇手」保護後仍有 1.9% NaN 級錯行,add-one 掃描再抓到 2 顆;名單隨卡/版本/校準變 |
| 不能按 scale 閾值 | PTv3 head 19 顆有 15 顆 subnormal,只有 6 顆出事;encoder 55 顆有 53 顆 subnormal 卻全對 |
| 線性島不需要沿鏈生長 | head 上 19 顆迷你島 1.077 ms vs 舊 region-island 1.087 vs 全 uniform 1.038(precision 0.998 = 純 fp16 天花板) |
| 海全 fp16 | 未量化區跑 fp32:CenterPoint 端到端 6.75 vs 4.44 ms(`three-model-results.md`) |

**cast 放哪**:只在「島↔海」與「圖 IO」邊界,每條跨界 float 邊恰好一顆;圖 IO 保 fp32
(runtime ABI)。**決定「這條邊要不要 cast」的是邊的 dtype,不是節點在不在島**:整數 / bool 邊
(zero-point、Reshape 的 shape)進出島都原封不動。dtype 來源依序是 `_ISLAND_FLOAT_INPUT_SLOTS`
(每個可進島的 op 一行,import 時 assert 齊全)、`onnx/dtypes.py::tensor_types`(圖 IO +
initializer + Constant + Q/DQ 輸出種子 + ONNX shape inference);兩者都說不出型別的邊 **raise**,
不猜 FLOAT。

**前置條件**:標準域 Q/DQ 要轉 fp16 需 opset ≥ 19(`QuantizeLinear` 的 fp16 `x`/`y_scale`);
圖低於 19 又有 conv 側 Q/DQ 時 pass 直接 raise,提示改 `deploy.onnx.opset_version`。
三個量化模型 config 已是 19。

**心智模型:島的 fp32 是「記號」不是執行精度。** TRT 把 `DQ→Gemm` 融成 int8 kernel、把海裡的
`Q/DQ→Conv→Relu→Q` 融成 int8 進出的 kernel;實際執行 = 海 fp16、量化 op int8、線性島邊界幾顆
cast。fp16-typed 線性 Q/DQ 的完整驗屍與重測工具:`work_dirs/reviews/fp16-typed-qdq-nogo.md`;
規則本身的實驗:`uniform-fp16-exception-rule.md`。

## 4. Verification:比對哲學

`verification/` 對 config 宣告的 scenario(如 `pytorch(cuda) vs tensorrt(cuda)`)
逐 tensor 比 max_diff。要點:

- **tolerance 是實測校準的,不是猜的**。量化/FP16 stage 的 raw-logit 跨 backend 差
  是預期行為(fake-quant vs 真 int8 kernel 的捨入路徑不同),**metric 相等才是真
  gate**。首跑 fail 時,錯誤訊息會給建議 gate(observed×1.25);把 observed 記進
  config 註解。
- 預設 tolerance 故意嚴,逼每個新模型做一次有意識的校準,而不是繼承一個形同虛設的
  大數字。

## 5. Evaluation:三 backend 全量 metric + latency

`deploy.evaluation` 用同一個 dataloader 對三個 backend 各跑一次完整 metric,輸出
並排(pytorch / onnx / tensorrt 三欄)。latency 表逐 stage 一行:看 `model_graphs`
(圖部分合計)評估量化/精度收益,看個別 stage 找瓶頸。ONNX Runtime 跑不了 plugin
stage(用 `torch_fallback_backends`)與 FP8 trt-domain op(該 experiment 直接關
onnx backend)。

## 6. 新增一個模型的 deploy 面

1. 在模型類實作 `build_stages()`:先全 TorchStage 跑通 pytorch backend,再逐段換成
   GraphStage。
2. experiment config 加 `deploy:` 區塊(參考 centerpoint / bevfusion / ptv3 現例):
   `onnx.precision`、`tensorrt.enabled`、verification scenarios、evaluation backends。
3. 先 `deploy deploy.tensorrt.enabled=false` 驗 onnx 正確性,再開 TRT。
4. verification 首跑 fail → 按 §4 校準 tolerance。
5. 有 sparse conv / 自訂 op → plugin 見 `docs/`(TRT plugin 建置)與
   `onnx_transforms` 現例(bevfusion sparse)。
6. 要量化 → 讀 [`../quantization/README.md`](../quantization/README.md) 的 checklist。

## 7. 檔案地圖

```text
deployment/
  stages.py        TorchStage / GraphStage / StageContext / validate_stages
  pipeline.py      StagedPipeline(三 backend 同一條)、PipelineCache、計時
  export.py        export 編排:export→modify_graph→precision→transforms→stamp→build
  onnx/
    export.py      torch.onnx.export 包裝
    precision.py   路由判定函式(custom domain / Q-DQ)、線性島 fp16 cast(§3)
    autocast.py    modelopt AutoCast 包裝、keep_topk_in_fp16
    modify.py      config 驅動的圖手術(deploy.onnx.modify_graph)
  backends/
    tensorrt_builder.py   strongly-typed build、plugin 載入
    tensorrt_runner.py    engine 執行
    onnx_runner.py        ORT 執行
  verification/
    backend_verifier.py   scenario 執行
    output_comparator.py  逐 tensor 比對、建議 gate
  config.py        deploy config schema
```

verify / evaluate 的**編排**不在這裡:`scripts/deploy.py` 依序呼叫 export → verify →
evaluate;latency 表與 metric 收斂在 `evaluation/evaluator.py`。

## 8. 深挖

- 量化(宣告、PTQ/QAT、INT8/FP8 選擇):`../quantization/README.md`
- fp16-typed Q/DQ NO-GO 全案(TRT NaN 缺陷、重測工具):`work_dirs/reviews/fp16-typed-qdq-nogo.md`
- 三模型量化交叉驗證數字:`work_dirs/reviews/` 下各 README
- 模型/訓練/資料面:`docs/contributing/adding-models.md`
