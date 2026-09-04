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
2. **export**:每個 GraphStage `torch.onnx.export`(opset 17),IO 名即宣告名。
3. **precision pass**(`onnx/precision.py`,自動路由,模型端零程式碼):

   | 圖的事實 | 走哪條 | 原因 |
   | --- | --- | --- |
   | 有自訂 domain(plugin) | 自家 island cast(整圖無島) | AutoCast 用 TRT parser 型別推導,不認 plugin op |
   | 有 Q/DQ(INT8/FP8) | 自家 island cast(fp32 島 + fp16 海) | AutoCast 拒收 Q/DQ 模型;island 是正確性地基,見 §3 |
   | 純圖 | modelopt AutoCast | 有數值守門(逐節點比對容差) |
   | `deploy.onnx.precision: fp32` | 原樣 | |

4. **TensorRT build**(`backends/tensorrt_builder.py`):**一律 strongly typed**——
   engine 的精度由 ONNX 圖的型別決定,不由 builder flag 猜。這是刻意決策:weak-typed
   加 `FP16` flag 會讓 TRT 的 kernel 自選精度,量化模型上曾實測翻車;strongly typed
   把精度變成**圖上可審查的事實**。plugin(`libautoware_tensorrt_plugins.so`)在
   build 前載入。
5. **verification**(§4)→ 6. **evaluation**(§5)。

## 3. Precision:fp16 的海、fp32 的島

量化圖的 fp16 化**不是**全圖轉型。Q/DQ 及其周邊保持 fp32-typed(「島」),其餘轉
fp16(「海」)。三層規則:

**誰進島**(`_quantized_island_names`,4 條依序):

1. 所有 Q/DQ 節點;
2. scale/zero-point 的 producer(fp32 scale 位元組級保留——scale 就是量化本身);
3. 每個 DQ 輸出的消費者(被量化的 Conv/Gemm 本體);
4. 反向生長:從每個 Q 的 data 輸入沿 **float data 邊**往回穿過 commuting whitelist
   (`Relu/Add/Concat/MaxPool/Reshape/Transpose/Gather/...`),讓「量化 op → pointwise
   → 下一個 Q」整段零 cast。

**cast 放哪**:只在「島↔海」與「圖 IO」邊界,每條跨界 float 邊恰好一顆;整數邊
(zero-point、shape、indices)永不 cast(`_ISLAND_FLOAT_INPUT_SLOTS` 顯式表 +
import 時 assert 與 whitelist 鎖死);圖 IO 保 fp32(runtime ABI)。

**為什麼**(每條都是量出來的):

| 規則 | 違反的實測代價 |
| --- | --- |
| scale 保 fp32 | fp16-typed Q/DQ 踩 TRT 10.8/10.16 缺陷:合併 scale subnormal → 融合 kernel 產 NaN、build 零警告(PTv3 mIoU 0.73→0.075) |
| DQ→消費者直連 | TRT INT8 融合 pattern 對不上,build assert |
| 鏈到下一個 Q 零 cast | Q-propagation 被 Cast 擋住 → 量化 conv 具現化 fp32:同一 backbone 4.76 vs 3.87 ms |
| 海全 fp16 | 未量化區跑 fp32:CenterPoint 6.75 vs 4.44 ms |

**最重要的心智模型:島的 fp32 是「記號」不是執行精度。** TRT 把島內
`DQ→Conv→Relu→Q` 融合成 int8 進出的 kernel;實際執行 = 海 fp16、島 int8、邊界幾顆
cast(實測合計 0.118 ms)。fp16-typed Q/DQ(opset 19 合法、ORT 算得對)在 TRT 上是
**NO-GO**,完整證據與重測工具:`work_dirs/reviews/fp16-typed-qdq-nogo.md`。

出現 `Quantized chain breaks at ...` 警告時:該 op 若量化可交換 → 加進
`_QDQ_COMMUTING_OPS`(連 slot 表一行,少一半 import 直接爆)並重跑三模型 battery;
不可交換(LayerNorm/Gelu 類)→ 加 `_KNOWN_NON_COMMUTING_OPS` 消音。

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
  export.py        deploy 流程編排(export→precision→build→verify→evaluate)
  onnx/
    export.py      torch.onnx.export 包裝
    precision.py   precision pass 路由、island 規則、commuting whitelist(§3 全部)
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

## 8. 深挖

- 量化(宣告、PTQ/QAT、INT8/FP8 選擇):`../quantization/README.md`
- fp16-typed Q/DQ NO-GO 全案(TRT NaN 缺陷、重測工具):`work_dirs/reviews/fp16-typed-qdq-nogo.md`
- 三模型量化交叉驗證數字:`work_dirs/reviews/` 下各 README
- 模型/訓練/資料面:`docs/contributing/adding-models.md`
