# Pruning 模組:結構化(channel)剪枝 + 知識蒸餾回收

> 讀者設定:第一次接觸這個框架的人。讀完你應該能:對 CenterPoint 跑一次寬度搜尋、
> 看懂 channel table、把剪過的 ckpt 直接丟給 `quantize` / `deploy`、以及替新模型接上剪枝。
> 量化面見 [`../quantization/README.md`](../quantization/README.md),部署面見
> [`../deployment/README.md`](../deployment/README.md)。

## 0. 三分鐘版

```bash
# 1) 搜尋 + KD fine-tune(可部署的路)
autoware-ml prune --config-name experiments/detection3d/centerpoint/<model>_pruned \
  --weights <fp_training.ckpt>
#    -> checkpoints/pruned.ckpt(搜尋完、未 fine-tune)與 best.ckpt / last.ckpt(fine-tune 後)

# 2) 之後的每一步都不需要 pruning config——ckpt 自己知道自己的架構
autoware-ml quantize --config-name .../<model>_pruned_int8 --weights <best.ckpt>
autoware-ml deploy   --config-name .../<model>_pruned_int8 --weights <ptq.ckpt>

# 只想量 latency(不訓練):
autoware-ml prune --config-name .../<model>_pruned --weights <fp.ckpt> \
  pruning.mode=search pruning.finetune=null
```

心智模型一句話:**模型宣告「哪一棵子樹可以剪」(架構事實,寫在 code),config 只給預算與
打分方式,搜尋結果是一張 channel table,存在 checkpoint 裡;`build_model` 先照表把架構
重建成窄的,再載權重。** 和 quantization 完全同一套設計。

## 1. 為什麼是 channel table 而不是改 config

FastNAS 是**逐層**搜寬度:同一個 block 裡相鄰的 conv 可以是 96→96→128→96。`SECONDBackbone`
的 `out_channels: [64, 128, 256]` 表達不了這件事,而且我們不想為了剪枝改每個 backbone / neck /
head 的建構參數。所以剪過的架構就是一張表:

```json
{"pts_backbone.blocks.1.2.conv": {"in_channels": 96, "out_channels": 128},
 "pts_backbone.blocks.1.2.norm": {"num_features": 128}, ...}
```

`apply_channel_table(model, table)` 對表裡每個模組用**同一個 class、同樣的超參數**(kernel /
stride / padding / eps / momentum …)重建一個新的,只換 channel 數。重建完的樹又全是普通的
`nn.Conv2d` / `nn.BatchNorm2d`,所以 BN fusion、Q/DQ 插入、ONNX export 一個字都不用改。

## 2. 三個階段

| 階段 | 誰做 | 產物 |
| --- | --- | --- |
| **search**(`pruning.mode: search`) | `search.py`:快取子樹的 stage 輸入 → ModelOpt FastNAS(`flops` 上限、`score` 打分)→ 就地縮窄 → `ChannelTable.record` | `pruned.ckpt` = state_dict + `pruning` payload |
| **finetune**(`pruning.mode: finetune`) | 同上,再用 Lightning 跑短程訓練:`PruningCallback` 讓每個存檔都自描述、`DistillationCallback` 把 KD 項加進 loss | `best.ckpt` / `last.ckpt` |
| **載入**(quantize / deploy / test) | `build_model` 偵測 payload → `apply_channel_table` → 驗證表 → 載權重 → (有 quantization payload 就再 `plan.prepare`) | — |

順序永遠是 **pruning → quantization**:先把架構弄對,量化計畫才在對的樹上跑。剪過再量化的
ckpt 兩個 payload 都帶(`save_quantized_checkpoint` 與 `QATCallback` 會順手寫入)。

## 3. 打分:`map` 還是 `proxy`

- `score: map`(預設):每個候選子網用 deploy 的 evaluate loop 算真 mAP(`score_samples` 張,
  CenterPoint 上約 2 s / 次)。搜出來的寬度是平台狀、heatmap branch 會被保住。**要拿去訓練用這個。**
- `score: proxy`:候選輸出對未剪網路輸出的相對 L2。幾秒搜完,但寬度鋸齒狀 → 只給 smoke 用。

兩個都要注意一件 FastNAS 的事:它對**每個**候選(含未剪的那個)都用快取的 frames 重校 BN,
所以任何自訂 score 的參考值必須在 convert 之後、同一個校準狀態下取(`search.py` 的 proxy
teacher 因此是 lazy 的)。參考值取錯的症狀:sensitivity 出現負值 → 整張 map 歸零 → 搜到最小子網。

## 4. 知識蒸餾怎麼接

`MultiTaskBaseModel` 有一個訓練期的 auxiliary-loss registry:

```python
pl_module.register_auxiliary_loss("kd", lambda inputs, outputs: {"loss_kd": ...})
```

`_core_step` 在 `compute_metrics` 之後把每一項加進 `loss` 並一起 log(只在 train)。
`DistillationCallback` 用它掛上 `kd_weight * model.distillation_loss(student_outputs,
teacher_outputs)`;teacher 是搜尋**前**的 FP 模型 deepcopy(同一份 `--weights`),凍結、eval。

`distillation_loss` 是模型的知識(CenterPoint:heatmap logits 對 teacher 機率做 BCE、回歸圖只在
teacher 看到物體的 cell 上做 L1),寫在 `main_modules/<model>/pruning.py`。

## 5. 替新模型接上剪枝

1. 寫一個 **fx 可追**的子樹 wrapper:forward 吃一顆 stage 輸入 tensor、回傳 tensor tuple、
   **持有模型自己的子模組(不能 deepcopy)**。容器 forward 追不過的(neck 迭代 list、head 建
   dataclass)就在 wrapper 裡 inline 呼叫子模組——被當成 leaf 的模組不可剪。
2. `build_pruning_spec()` 回傳 `PruningSpec(subtree, stage_input, submodules)`。
3. 要 KD 就實作 `distillation_loss(student, teacher)`。
4. 一個 `<model>_pruned.yaml`。

不用改 engine。spconv / 自訂 op 的層不在 ModelOpt registry 裡,會自然變成 leaf(不剪、不壞)。

## 6. 已知邊界

- FastNAS **只剪寬度**,不剪深度(`layer_nums` / `enc_depths` 這維度碰不到)。
- 需要 `torchprofile==0.0.4`(ModelOpt 的 FLOPs 計數用它的舊 API)。
- 單卡:teacher 是 strategy wrap 外的普通模組;callback 會拒絕多卡。
- 剪過的模型與原 release 模型**不再是同一張網**:所有 parity 對照(AWML ↔ autoware-ml、
  bit-exact)對它不成立;`skip_quantize` 之類的量化 recipe 要在剪過的樹上重掃。
