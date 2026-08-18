# StreamPETR 畢業路線 — DGX Station A100 完整執行 Runbook

適用機器:`tieriv-DGX-Station-A100-920-23487-2531-0R0`
(4× A100-SXM4-80GB = GPU 0,1,2,4;**GPU 3 是 DGX Display,絕對不能進訓練**;driver 570.133.20,CUDA 12.8 相容)

所有指令 2026-08-10 已在工作站(2× RTX PRO 6000)以 smoke test 驗證過流程;
本文件把 batch/LR 換算成 DGX 的 4-GPU 配置。

---

## Batch / LR 換算原則(先讀這段)

三個 stage 的 config 都把 **總 batch 釘在 16**、LR 依此寫死(repo 沒有 auto_scale_lr)。
本 runbook 選擇 **4 GPU × `batch_size=8` = 總 batch 32** 求快,LR 依 AWML 自己的線性換算慣例
(`total_batch/8 × base`,config 註解:`bs 8 -> 2e-4, bs 16 -> 4e-4`)乘 `N/16 = 2`:

| Stage | config 預設(總 batch 16) | DGX 覆寫(總 batch 32) | LR 覆寫(×2) |
|---|---|---|---|
| 1 nuScenes | 2 GPU × 8,lr 4e-4 / backbone 4e-5 | `batch_size=8 trainer.devices=4` | `model.optimizer.lr=8.0e-4` / `...img_backbone.lr=8.0e-5` |
| 2 T4 base | 2 GPU × 8,lr 1e-4 / backbone 1e-5 | `batch_size=8 trainer.devices=4` | `model.optimizer.lr=2.0e-4` / `...img_backbone.lr=2.0e-5` |
| 3 j6gen2 finetune | 同 stage 2 | `batch_size=8 trainer.devices=4` | 同 stage 2 |

每卡 batch 8 = 原始 2-GPU 配方的每卡量,80GB A100 上驗證過,記憶體不是問題(前提是卡空)。

⚠️ **穩定性風險集中在 stage 1**:AdamW 8e-4 已偏高,且 stage 1 的 `gradient_clip_val: 35` 很鬆。
開跑後**盯前幾百個 iteration 的 loss**,出現 spike/NaN 就把兩個 LR 退到 6e-4 / 6e-5 重跑。
warmup(500 iter,step-based)不用動。

> 保守備案:若要嚴格重現已驗證配方,退回 `batch_size=4 trainer.devices=4`(= 16)且 **LR 全部不覆寫**
> ——那正好等於 AWML 原始 nuScenes recipe(`num_gpus=4, batch_size=4, lr 4e-4`)。

⚠️ **GPU 現況**:nvidia-smi 顯示 4 張 A100 各已被其他 job 佔 45–50 GB / util 95–100%。
batch 4 都在剩餘 ~30 GB 邊緣,batch 8 需求更高,大概率塞不下且會互搶頻寬。
**等卡空了再跑正式訓練**,
smoke test 可以先做。

---

## 0. 前置:程式碼、image、資料

```bash
# 0-1. 程式碼(帶 streampetr branch 的 autoware-ml)
git clone <你的 autoware-ml repo> ~/autoware-ml
cd ~/autoware-ml && git checkout <streampetr-branch>

# 0-2. Docker image(需要 ghcr 權限:docker login ghcr.io)
docker pull ghcr.io/tier4/autoware-ml:latest

# 0-3. 資料。DGX 上同樣掛著 QNAP(/mnt/qnapdata/...),**不用 rsync**,
#      直接 bind-mount 進 container(見 §1)。先確認路徑和內容都在:
ls /mnt/qnapdata/external/nuscenes/          # 需要: samples/ sweeps/ v1.0-trainval/
                                             #       nuscenes_infos_{train,val,test}.pkl
ls /mnt/qnapdata/internal/t4datasets/        # 需要: 各 DB 目錄(sensor data,infos 內相對路徑引用)
ls /mnt/qnapdata/internal/t4datasets/info/detection3d/   # t4dataset_base_infos_{train,val,test}.pkl (stage 2)
ls /mnt/qnapdata/internal/t4datasets/info/kokseang_2_8/  # t4dataset_j6gen2_base_infos_{train,val,test}.pkl (stage 3)
```

> **IO 備案**:QNAP 是網路儲存,StreamPETR 每個 sample 要讀 6 張相機圖,若訓練時
> data loading 成瓶頸(GPU util 低、`num_workers` 吃滿仍餵不飽),再把 nuScenes
> rsync 到 DGX 本地 NVMe(`rsync -avP /mnt/qnapdata/external/nuscenes/ $HOME/data/nuscenes/`)
> 並把 mount source 換掉。第一次先直接用 QNAP 跑 smoke 看速度。

## 1. 起 container

```bash
# 用 UUID 選卡,不用 index:nvidia-container-cli 的裝置編號跟 nvidia-smi 不一致
# (DGX 實測:--gpus '"device=0,1,2,4"' 會噴 device error: 4: unknown device)
A100S=$(nvidia-smi -L | grep A100 | sed -E 's/.*UUID: (GPU-[0-9a-f-]+)\).*/\1/' | paste -sd,)
echo "$A100S"   # 必須恰好 4 個 GPU-xxxx,DGX Display 已被 grep 排除

docker run -d --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus "\"device=$A100S\"" \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e AUTOWARE_ML_DATA_PATH=/workspace/data \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  --mount type=bind,source=$HOME/autoware-ml,target=/workspace \
  --mount type=bind,source=/mnt/qnapdata/external/nuscenes,target=/workspace/data/nuscenes \
  --mount type=bind,source=/mnt/qnapdata/internal/t4datasets,target=/workspace/data/t4datasets \
  --name autoware-ml-vivid ghcr.io/tier4/autoware-ml:latest sleep infinity

# 進 container(-l 必要:pixi 環境靠 login shell 進 PATH)
docker exec -it autoware-ml-vivid bash -l

# ★ 開跑前必查(container 內):
nvidia-smi -L          # 必須恰好 4× A100,沒有 "DGX Display"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
                       # 必須 True 4;若 False → 見文末坑 1
```

`--gpus "device=<UUID,...>"` 用 UUID 排除 Display GPU;container 內會重新編成 cuda:0–3,
config 的 `trainer.devices=4` 直接可用。

## 2. Phase 0:backbone 下載 + RGB 轉換(container 內,一次性)

```bash
cd /workspace
mkdir -p work_dirs/ckpts

# upstream StreamPETR 公開發布的 DD3D/FCOS3D VoVNet-99(292 MB;container 沒有 wget,用 curl)
curl -L -o work_dirs/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth \
  https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth
md5sum work_dirs/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth
# 應為 ff1ac3040eabf0f0e54c3c594c26021e

# BGR→RGB stem 翻轉(預期輸出:Converted 626 tensors; skipped 81)
python -m autoware_ml.tools.convert_streampetr_checkpoint \
  --input  work_dirs/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth \
  --output work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
  --bgr-to-rgb
```

## 3. Smoke test(每個 stage 各 1 分鐘級,正式跑之前必做)

```bash
cd /workspace

# Stage 1 smoke
autoware-ml train --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
  --weights work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
  +trainer.fast_dev_run=true trainer.devices=1
# 過關訊號:log 有「Loaded matching weight tensors: 626/1526 (+626 shared-tensor aliases)」、exit 0

# Stage 2 smoke(--weights 先隨便用 stage 1 smoke 不會產 ckpt,這裡直接用 backbone 驗流程)
autoware-ml train --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
  --weights work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
  datamodule.data_root=/workspace/data/t4datasets \
  +trainer.fast_dev_run=true trainer.devices=1

# Stage 3 smoke
autoware-ml train --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune \
  --weights work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
  datamodule.data_root=/workspace/data/t4datasets \
  datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl \
  +trainer.fast_dev_run=true trainer.devices=1
```

預期載入訊號:stage 1/2/3 smoke 用 backbone 權重都是「626/1526 (+626 aliases)」。
正式訓練時 stage 2/3 吃前一 stage 的 checkpoint,會是「**1526/1526 (+0 aliases)**」全覆蓋
(2026-08-10 工作站實測:三個 stage 的 fast_dev_run 全 exit 0,stage 間 checkpoint 交接 1526/1526)。

## 4. Stage 1 — nuScenes pretrain 正式訓練(30 ep,4×A100 batch 32;**獨占卡實測約 8 小時**)

```bash
cd /workspace
nohup autoware-ml train \
  --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
  --weights work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
  batch_size=8 trainer.devices=4 \
  model.optimizer.lr=8.0e-4 \
  model.optimizer_group_overrides.img_backbone.lr=8.0e-5 \
  > work_dirs/stage1_pretrain.log 2>&1 &

tail -f work_dirs/stage1_pretrain.log
```

- checkpoint 落在 `mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/{best,last}.ckpt`
  (best = val/det3d/mAP 最高;validation 每 5 epoch 跑一次)。
- **早期熔斷**:前 2–3 epoch 對照 AWML model-zoo 的 logs
  (https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/streampetr/streampetr-vov99/nuscenes/v1.0/logs.zip),
  loss 走勢明顯漂就停下來查。注意 reference logs 是總 batch 16 跑的,batch 32 + LR ×2 下
  逐 iteration 數值不會對齊,只比走勢(有無發散/spike);另外 8e-4 + grad-clip 35 偏激進,
  **前幾百 iter 出現 loss spike/NaN → LR 退 6e-4 / 6e-5 重跑**(見 §Batch/LR 換算原則)。
- 及格線:7-class val mAP 對齊 acceptance bar(見計畫 Phase 0-4;10-class 原始公布值 mAP 0.4697)。

**2026-08-13 實測結果(run `5ef20538`,DGX 4×A100 batch 32 / lr 8e-4)**:

| epoch | 9 | 14 | 19 | 24 | **29(final)** |
|---|---|---|---|---|---|
| val/det3d/mAP | 0.2699 | 0.3346 | 0.4010 | 0.4415 | **0.5031** |

單調上升、無 spike/NaN——8e-4 + grad-clip 35 這組在 stage 1 實際上是穩的。
best.ckpt = epoch 29(0.50314),直接餵給 stage 2。

⚠️ **速度會被別人的 job 嚴重拖累**:同一台機器上有另一個 4-GPU job 並存時實測
**65 分鐘/epoch**;獨占 4 張卡時是 **15 分鐘/epoch**(4 倍差)。排程前先 `nvidia-smi`
看有沒有別人在跑,並且**不要**看到「卡是空的」就隔天才送出指令——送出前要重新確認
(踩過:憑一天前的觀察 resume,撞上別人新起的 job,3 個 rank OOM 後整個 DDP deadlock)。

### MLflow 監控

metrics 全部寫進 `mlruns/mlflow.db`(SQLite),repo 內建 UI 指令:

```bash
# container 內起 UI(--net=host,直接綁 DGX 的 0.0.0.0:5000)
docker exec -d autoware-ml-vivid bash -lc \
  'cd /workspace && autoware-ml mlflow ui > work_dirs/mlflow_ui.log 2>&1'
```

- 筆電看:VSCode Remote-SSH 會自動轉發 port 5000 → http://localhost:5000;
  或手動 `ssh -L 5000:localhost:5000 <dgx>`。
- 盯的指標:`train/loss`(前幾百 iter 有無 spike/NaN)、`lr-AdamW`(warmup 走勢)、
  每 5 epoch 的 `val/det3d/mAP`;checkpoint 在 run 的 Artifacts → `checkpoints/{best,last}.ckpt`。

### 暫停 / Resume(三個 stage 通用)

**暫停** = 殺掉訓練 process。`last.ckpt` 每個 epoch 結尾更新,所以最多丟掉當前跑到一半的 epoch:

```bash
# container 內:送 SIGINT 給所有訓練 process(Lightning 會走 KeyboardInterrupt 收尾)
pkill -INT -f vov_320x800_nuscenes_pretrain      # ← 換成當前 stage 的 config 名

# 等 ~30s 後確認死透(GPU 記憶體歸零);殘留就補 pkill -9 -f 同 pattern
nvidia-smi
```

**恢復** = `--resume-checkpoint` 指向該 run 的 `last.ckpt`(還原 weights、optimizer state、epoch,
**metrics 續寫進原本的 MLflow run**,曲線不斷線):

```bash
cd /workspace
LAST=mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/last.ckpt

nohup autoware-ml train \
  --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
  --resume-checkpoint $LAST \
  batch_size=8 trainer.devices=4 \
  model.optimizer.lr=8.0e-4 \
  model.optimizer_group_overrides.img_backbone.lr=8.0e-5 \
  > work_dirs/stage1_pretrain_resume.log 2>&1 &
```

規則:

- `--resume-checkpoint` 和 `--weights` **互斥**——resume 時把 `--weights` 拿掉。
- 其他 override(`batch_size`、`trainer.devices`、兩個 LR、stage 2/3 的 `datamodule.*`)
  **必須原封不動重打**,config 不會從 checkpoint 還原。
- 想把後續訓練分岔到新的 MLflow run(例如改了 LR 重跑)就加 `--new-run`;
  單純斷點續跑**不要**加。
- stage 2/3 同理,把 config 名、`$LAST` 路徑和各自的 override 換掉即可。

## 5. Stage 2 — T4 base DB(35 ep)

```bash
cd /workspace
S1=mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt
# S1=/workspace/mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/5ef20538051d405d92ffc7ba1296dc46/artifacts/checkpoints/best.ckpt

nohup autoware-ml train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
  --weights $S1 \
  datamodule.data_root=/workspace/data/t4datasets \
  batch_size=8 trainer.devices=4 \
  model.optimizer.lr=2.0e-4 \
  model.optimizer_group_overrides.img_backbone.lr=2.0e-5 \
  > work_dirs/stage2_t4base.log 2>&1 &
```

stage 間零轉檔:`--weights` 直接吃 Lightning checkpoint(7-class pretrain → head 完整載入,無 drop)。

**2026-08-15 實測結果(run `56cdf6a3`,4×A100 batch 32 / lr 2e-4)**

跑 35 epoch 花 **40 小時**(2026-08-13 11:54 → 08-15 03:53,約 69 分鐘/epoch)。
比 stage 1 的 15 分鐘/epoch 慢 4.6 倍,因為每 epoch 實餵 111,840 frames(nuScenes 是 26,944,4.15 倍)。

| validation | 1 | 2 | 3 | 4 | 5 | 6 | **7(ep34)** |
|---|---|---|---|---|---|---|---|
| val/det3d/mAP | 0.2663 | 0.2973 | 0.3748 | 0.3760 | 0.4062 | 0.4337 | **0.4420** |

best.ckpt = epoch 34(0.44203);train/loss 22.93 → 12.75,無 spike。

最終 per-class 與距離分佈:

| car | bus | truck | pedestrian | bicycle | | 0-50m | 50-90m |
|---|---|---|---|---|---|---|---|
| 0.5977 | 0.5410 | 0.4401 | 0.3687 | 0.2626 | | 0.4718 | 0.1611 |

⚠️ **base DB 只評到 5 類**:`barrier` 和 `traffic_cone` 完全沒有 metric,代表 base 的
`t4dataset_base_infos_*` 沒有這兩類標註。config 宣告 7 類、head 也是 7 類,但這兩顆輸出在
stage 2 的 35 epoch 是監督空窗(stage 1 在 nuScenes 學到的 barrier 0.4955 /
traffic_cone 0.5632 會因此退化,靠 stage 3 找回)。

stage 3 的 j6gen2 config **有訓練這兩類**:7 類全訓 + `partial_ignore_classes:
[traffic_cone, barrier]`(沒標這兩類的 frame 不把背景預測當 FP 懲罰,有標的照常訓練)。
最終這兩類偏低(test cone 0.339 / barrier 0.290)的主因就是 stage 2 空窗 + j6gen2 僅部分
scene 有標。要拉分數,方向是給 stage 2 換成帶這兩類標註的 base info,而不是在 stage 3 之後
再多 finetune 一輪。
(注:`vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml` **不是**「補練這兩類」的
config——它與 `_finetune` 同類別、同 partial-ignore,只是舊轉檔路線用的單卡低 LR 變體,
檔名沿用 AWML 原始 config 名。)

## 6. Stage 3 — j6gen2 finetune(35 ep)+ 最終驗收

```bash
cd /workspace
S2=mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt

nohup autoware-ml train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune \
  --weights $S2 \
  datamodule.data_root=/workspace/data/t4datasets \
  datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl \
  batch_size=8 trainer.devices=4 \
  model.optimizer.lr=2.0e-4 \
  model.optimizer_group_overrides.img_backbone.lr=2.0e-5 \
  > work_dirs/stage3_j6gen2.log 2>&1 &

# 訓完評測(單卡;headline 是 0-121m bucket mAP)
S3=mlruns/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune/<run_id>/artifacts/checkpoints/best.ckpt
autoware-ml test \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune \
  --weights $S3 \
  datamodule.data_root=/workspace/data/t4datasets \
  datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl \
  trainer.devices=1
```

**驗收表**(±1 mAP 內宣告畢業):

| 路線 | val mAP | test mAP | checkpoint |
|---|---|---|---|
| 轉檔路線(autoware-ml, 92068f7b) | 0.39127 | 0.36609 | — |
| AWML aligned_bf16 | 0.37521 | 0.35515 | — |
| native 畢業路線 batch 32(run `977ecb06`) | 0.51775 | 0.49810 | **已誤刪,數字僅供參考** |
| **native 畢業路線 batch 64(run `55884dc0`,重跑)** | **0.51604** | **0.50246** | ✅ `~/ckpt_backup/` 有備份 |

> **2026-08-17 注記**:run `977ecb06` 的 checkpoint 事後被誤刪,以下數字仍有效
> (MLflow 記錄在)但 ckpt 不可復得。重跑改用 **batch 16×4 = 64、lr 4.0e-4 / 4.0e-5**
> (run `55884dc0`,使用者決定)——實測 throughput 只比 batch 32 快 6.5%
> (26.2 → 27.9 frames/s,瓶頸在 QNAP IO,約 28 frames/s 天花板;記憶體 ~52 GB/卡)。
> 新配方之後,兩個 LR 是未驗證組合,結果不可與下表直接混用。

**2026-08-16 實測結果(train run `977ecb06`,test run `0f399455`)**

35 epoch 花 22 小時(08-15 14:00 → 08-16 11:55,約 37.5 分鐘/epoch);
best = epoch 34;train/loss 14.27 → 10.90,無 spike。
sampler 每 epoch 實餵 52,032/56,436 frames(4 rank × 8 lane 修掉 7.8%)。

val mAP 歷次(每 5 epoch):0.3633 → 0.4403 → 0.4259 → 0.4740 → 0.4866 → 0.5167 → **0.5178**

test set(headline = 0-121m):

| 指標 | 0-121m | 0-50m | 50-90m | 90-121m |
|---|---|---|---|---|
| mAP | **0.4981** | 0.5303 | 0.1558 | 0.0000 |
| mAPH | 0.4428 | 0.4739 | 0.1280 | 0.0000 |
| mAP-based NDS | 0.5715 | 0.5883 | 0.3359 | 0.0000 |

test per-class mAP(0-121m):

| car | bus | truck | bicycle | pedestrian | traffic_cone | barrier |
|---|---|---|---|---|---|---|
| 0.6735 | 0.6571 | 0.5937 | 0.4701 | 0.4632 | 0.3392 | 0.2900 |

**大幅超過驗收線(val +12.6 / test +13.2)而非落在 ±1 內。** 注意這不是同配方對照:
本路線用 batch 32 + LR ×2(舊路線是 batch 16),且 stage 1/2 都是 native 全新訓練
(stage 1 nuScenes 0.5031、stage 2 base 0.4420),與轉檔路線的 upstream 權重來源不同。
評測協定本身一致(同 config、同 kokseang_2_8 split、同 min_num_points 對齊)。
90-121m 全為 0 是模型設計使然:`point_cloud_range ±51.2m`、`post_center_range ±61.2m`
(對角最遠 ~86m),90m 外根本不會有預測——同 config 的路線都一樣,不是 bug。

**2026-08-18 重跑實測(train run `55884dc0`,test run `80383672`;batch 16×4=64、lr 4e-4/4e-5)**

35 epoch 花 29 小時(08-17 02:57 → 08-18 07:53 JST;含中途 GPU 分時波動);
best = epoch 34,val mAP **0.51604**(vs batch 32 的 0.51775,−0.17 pp,±1 判準內)。
val 軌跡前期明顯落後 batch 32(ep4 −4.5 pp、ep9 −5.1 pp),cosine 收尾追平——
印證「大 batch + 等比 LR 在 finetune 段前期學習較糙」,最終品質等價但**沒有比較快**
(throughput 差 6.5%,IO-bound),下次重現建議回 batch 32 配方。

test set(headline = 0-121m):

| 指標 | 0-121m | 0-50m | 50-90m |
|---|---|---|---|
| mAP | **0.5025** | 0.5336 | 0.1730 |
| mAPH | 0.4468 | 0.4776 | 0.1388 |
| mAP-based NDS | 0.5670 | 0.5824 | 0.3401 |

test per-class mAP(0-121m):

| car | bus | truck | bicycle | pedestrian | traffic_cone | barrier |
|---|---|---|---|---|---|---|
| 0.6773 | 0.6511 | 0.5851 | 0.4564 | 0.4638 | 0.3436 | 0.3402 |

與 batch 32 run 逐類相差 ≤1.2 pp(barrier +5.0 pp 是最大單項差,方向偏好);
整體 test +0.44 pp。**三個 stage 的 best.ckpt 均已備份到 DGX `~/ckpt_backup/`。**

---

## 共用機器:查 GPU 被誰佔用

DGX 是共用的,開跑前先看有沒有別人在跑:

```bash
tools/whos_on_gpu.sh          # ★ 必須在 host 跑,不是 container 內
```

輸出每個 GPU process 的 host PID、佔用量、**所屬 container**、實際指令,以及各卡使用量。

⚠️ **container 內的 `nvidia-smi` 看不到 process list**(Processes 區塊會是空的)——
PID namespace 隔離,只有 host 看得到。所以「記憶體被吃掉但看不到誰在用」是正常現象,不是壞掉。

腳本是**可攜的**——任何有 nvidia-smi 的 Linux 機器都能跑,不需要 docker
(在 container 裡的 process 顯示 container 名字,在 host 上的顯示 Linux 使用者)。
沒辦法放檔案的機器就貼這個單行版:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader |
  while IFS=', ' read -r p m _; do
    printf '%-8s %-9s %s\n' "$p" "$m" "$(ps -o user=,etime=,cmd= -p "$p" 2>/dev/null | cut -c1-75)"
  done
```

腳本背後的映射鏈,手動查時可以照走:

```bash
# 1. 誰在吃 GPU(host PID + 用量)
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# 2. PID → 使用者(一般機器問到這裡就夠了)
ps -o user=,etime=,cmd= -p <PID>

# 3. 若是跑在 container 裡,user 會顯示 root — 再往下查 container 短 ID
grep -oE '[0-9a-f]{64}|docker[-/][0-9a-f]{12}' /proc/<PID>/cgroup

# 4. container ID → 名字 → 在跑什麼
docker inspect <短ID> --format '{{.Name}}'
docker exec <名字> bash -lc 'ps -eo pid,etime,cmd --sort=start_time | grep python'
```

OOM 的錯誤訊息本身也會直接點名 PID(`Process 3103693 has 72.14 GiB memory in use`),
拿那個 PID 從第 2 步接下去即可。

**借卡禮儀**:要停別人的 job 前,(a) 先取得對方同意,(b) 查他的 checkpoint 節奏
(mmdet3d 只在 epoch 結尾存檔),等他存完再停,對方就零損失,(c) 用 `pkill -INT` 讓框架
正常收尾,不要一上來就 `-9`。

### container 無故消失時怎麼判斷死因

```bash
docker inspect <名字> --format 'ExitCode: {{.State.ExitCode}}
OOMKilled: {{.State.OOMKilled}}
FinishedAt: {{.State.FinishedAt}}'
docker events --since 12h --until 1s --filter type=container \
  --format '{{.Time}} {{.Actor.Attributes.name}} {{.Action}}'
last -n 10          # 同時段誰登入過
```

判讀:

| 徵狀 | 死因 |
|---|---|
| `ExitCode 137` + `OOMKilled: true` | container 記憶體上限撞到 |
| `ExitCode 137` + `OOMKilled: false` + events 有 **`stop`** | **有人手動 `docker stop`/`kill`** |
| `ExitCode 137` + dmesg 有 `oom-kill` | host RAM 被吃光,kernel OOM killer |
| `ExitCode 0` | PID 1 自己正常結束 |

`stop` 這個 event 只有人下指令才會產生,程式自己崩潰不會有——**2026-08-13 11:34 實際踩過一次**
(stage 2 開跑 39 分鐘後被人停掉,當時尚未寫出任何 checkpoint,整段重跑)。
共用機器上跑長工前,建議先在群組講一聲。

## 已知坑(工作站實測踩過,DGX 上照查)

1. **torch CUDA init 失敗但 nvidia-smi 正常** → container 缺 `/dev/nvidia-uvm`;
   啟動命令已含 `--device /dev/nvidia-uvm{,-tools}`,若仍 False 先確認 host `ls /dev/nvidia-uvm*`。
2. **短測(非 fast_dev_run)必加 `trainer.check_val_every_n_epoch=1`**:
   config 預設每 5 epoch 才 validate,`max_epochs=1` 的短測不會跑 validation,
   train.py 收尾查 `val/loss` 直接 ValueError。正式訓練不受影響。
3. **GPU 3 = DGX Display(4 GB)**:混進 DDP 會 OOM + NCCL 掛掉,用 UUID 選卡排除(見 §1)。
   注意 **不能用 nvidia-smi 的 index**(`--gpus '"device=0,1,2,4"'`):nvidia-container-cli
   的編號跟 nvidia-smi 不一致,DGX 實測噴 `device error: 4: unknown device`。
4. Lightning 沒有 AWML 的 `dynamic_intervals`:最後 5 epoch 不會逐 epoch validate(已知差異,可接受)。
5. `nohup` 之外也可用 tmux;斷線後 `docker exec -it awml-streampetr bash -l` 回去 `tail -f` 即可。
