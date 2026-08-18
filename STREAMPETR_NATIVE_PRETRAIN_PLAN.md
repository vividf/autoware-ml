# StreamPETR Native Pretrain — 畢業計畫(tracking 用,未追蹤、不進 commit)

**目標**:nuScenes pretrain 在 autoware-ml 原生重訓,讓三階段流程(nuScenes → T4 base → j6gen2)全程 native、全程 RGB,
從此不再依賴 AWML model-zoo checkpoint 轉檔。convert tool 降級為「FCOS3D backbone init 的一次性步驟」。

**前提確認(2026-08-10)**:
- autoware-ml pipeline 天生 RGB-native:`transforms/camera/loading.py` 讀圖即 `COLOR_BGR2RGB`,
  `streampetr/base.yaml` 的 `img_norm_cfg` 是 RGB 序 stats + `to_rgb: false`。原生重訓不需要「改成 RGB」。
- BGR 唯一殘留 = FCOS3D backbone init(mmcv 慣例 BGR 訓練),`convert_streampetr_checkpoint --bgr-to-rgb`
  翻 stem conv 一次即永久解決。
- Stage 1 配方已存在:`configs/tasks/detection3d/streampetr/vov_320x800_nuscenes_pretrain.yaml`
  (CPFPN + FocalHead2D、lr 4e-4、grad-clip 35、2 GPU × batch 8 = 16、30 ep、flip + rot/scale 增強)。
- Stage 2/3 配方已存在且 stage 間 `--weights` 直接吃 Lightning checkpoint,零轉檔。

**與 AWML pretrain 的已知差異(有意保留,不是 bug)**:
- **7-class vs 10-class**:autoware-ml nuScenes 是 7 class(name_mapping 把 construction_vehicle 等併入),
  AWML pretrain 是 10 class。紅利:stage 2 載入不需要 `--drop-pattern`,classification head 完整保留。
  代價:nuScenes mAP 與 AWML 公布的 0.4697(10-class)不可直接比 → 見 Phase 0-3 的 bar 建法。
- Lightning 無 `dynamic_intervals`:最後 5 個 epoch 不會逐 epoch validate(docs 已記載)。
- 這**不是 bit-parity 重現**(AMP 實作、sampler 細節不同)。數字對不上時的歸因工具是
  Phase 0 的 bar + AWML logs 曲線,不是 tensor 比對。最終仲裁在 Phase 3 的 T4 數字。

---

## Phase 0 — 一次性準備(~半天)

- [x] **0-1. 取得 FCOS3D backbone(主路徑:upstream 公開 artifact)**。✅ 2026-08-10 驗證完成:
      upstream 直接發布的就是 `-remapped` 版,GitHub release 直連可 wget(292 MB):
      ```bash
      wget -P work_dirs/ckpts/ \
          https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth
      # md5: ff1ac3040eabf0f0e54c3c594c26021e
      ```
      實測結論:
      - 檔案是**裸 state_dict**(無 wrapper),命名已是 `img_backbone.stem.stem_1/conv.weight`
        的 mm layout,native VoVNet 鏡射命名直接吃 → **remap 完全不需要,convert tool 不用加任何選項**。
      - 內含 `img_backbone.*` 626 tensors + `bbox_head.*` 81 tensors(FCOS3D 自己的 2D head);
        後者被 converter 的 prefix 規則自動 skip。
      - AWML 內部那顆本機檔在 host 和 awml container 都不存在,交叉驗證跳過(upstream 出處已足夠)。
- [x] **0-2. 最後一次 convert(RGB flip)**。✅ 2026-08-10 驗證完成(container 內執行):
      ```bash
      python -m autoware_ml.tools.convert_streampetr_checkpoint \
          --input work_dirs/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth \
          --output work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth \
          --bgr-to-rgb
      # → Converted 626 tensors; skipped 81 (全部是 bbox_head.*)
      ```
      驗證結果(兩層都過):
      - stem 翻轉等價性:BGR 像素餵原權重 vs RGB 像素餵翻轉權重,stage4/stage5 feature
        `allclose(atol=1e-4)` 通過,相對 L2 誤差 ~1e-6。**注意:不是 bitwise 相等** —
        channel 翻轉改變卷積在 channel 維的浮點加總順序,fp32 噪音級差異是預期行為,
        驗收準則用 allclose(rel ~1e-6),不要用 bitwise。
      - 命名完整性:① `VoVNet99MultiScale.load_state_dict(strict=True)` 全覆蓋通過;
        ② hydra 組 `vov_320x800_nuscenes_pretrain` config 實例化整模 + `apply_matching_weights`
        (與 `autoware-ml train --weights` 同一路徑)→ 1252 tensors 載入
        (`img_backbone.*` 626 + `image_feature_extractor.*` alias 626),零 shape mismatch。
- [x] **0-3. nuScenes 資料就緒**。✅ 2026-08-10 確認:`/mnt/qnapdata/external/nuscenes/` 下
      `nuscenes_infos_{train,val,test}.pkl` 已存在(mmdet3d v1.1 schema,val 6,019 筆正確),
      datamodule 直接可讀 — 已由實際 smoke training 驗證(見「執行環境」段的掛載方式)。
- [ ] **0-4. 建立 acceptance bar(inference-stack 驗證,順便得到 Phase 1 及格線)**:
      把已轉檔的 model-zoo pretrain(10-class,不加 drop-pattern)在 autoware-ml 用
      **10-class dataset override**(class_names = AWML 10 類、name_mapping identity)跑 nuScenes eval。
      - 對照 AWML 公布值 mAP 0.4697 / NDS 0.5568(eval range 50 m)→ 驗證 nuScenes 端 eval + inference
        stack 對齊(T4 端經驗:same-weights residual 約 −0.8 mAP,屬 camera pipeline 數值差,可接受)。
      - 記下 10 類的 per-class AP(car 0.636 / pedestrian 0.541 / traffic_cone 0.646 / barrier 0.607 …)
        → 投影到 7 個共同 class 作為 Phase 1 的 per-class bar。

## 執行環境(2026-08-10 實測可用)

nuScenes 在 host 的 `/mnt/qnapdata/external/nuscenes`(infos pkl 已就位:
`nuscenes_infos_{train,val,test}.pkl`,mmdet3d v1.1 schema,datamodule 直接可讀)。
既有 `autoware-ml-yihsiang` container 只掛了 internal,另起一個掛 nuScenes 的 container:

```bash
docker run -d --net=host --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e AUTOWARE_ML_DATA_PATH=/workspace/data \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  --mount type=bind,source=/home/yihsiang/autoware-ml,target=/workspace \
  --mount type=bind,source=/mnt/qnapdata/external/nuscenes,target=/workspace/data/nuscenes \
  --name autoware-ml-yihsiang-nusc ghcr.io/tier4/autoware-ml:latest sleep infinity

docker exec -it autoware-ml-yihsiang-nusc bash -l   # 進去跑 autoware-ml
```

⚠️ 已踩坑:不加 `--device /dev/nvidia-uvm` 時,container 內 `nvidia-smi` 正常但 torch CUDA
init 失敗(toolkit 漏掛 uvm node)。另注意 2026-08-10 當下**舊 container `autoware-ml-yihsiang`
的 torch CUDA 也處於壞掉狀態**(host cuInit 正常)— 如果要在舊 container 訓練要先重啟它。

## Phase 1 — nuScenes pretrain 原生重訓(主要成本;2 GPU 約 4 天)

- [ ] **1-1. 開跑前 checklist**:
  - [ ] AMP:AWML 用 dynamic-scale AMP(`NoCacheAmpOptimWrapper`)。確認 trainer precision
        (bf16 經驗已有,擇一並記錄在本文件)。
  - [ ] camera order:pretrain 階段**固定順序**(shuffle 是 T4 配方);確認繼承鏈沒帶進 shuffle。
  - [ ] `check_val_every_n_epoch: 5`、`devices: 2` 沒被 CLI override 破壞(lr 4e-4 綁定總 batch 16)。
- [ ] **1-2. 訓練**(container 啟動方式見下方「執行環境」段):
      ```bash
      autoware-ml train \
          --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
          --weights work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth
      # data_root 預設 ${AUTOWARE_ML_DATA_PATH}/nuscenes = /workspace/data/nuscenes,掛對就不用 override
      ```
      短測注意:`fast_dev_run=true` 可用;但若改用 `limit_train_batches` + `max_epochs=1` 做短測,
      **必須加 `trainer.check_val_every_n_epoch=1`** — 否則 validation 不會跑(config 預設每 5 epoch),
      train.py 收尾的 optimized-metric(`val/loss`)查不到會直接 ValueError(已實測踩過)。
- [ ] **1-3. 早期熔斷**:model-zoo `logs.zip`(AWML 訓練日誌)先下載;前 2–3 epoch 比 loss 曲線,
      量級/走勢明顯漂移就停下來查,不要燒滿 4 天。
- [ ] **1-4. 及格標準**:
  - 7-class val mAP ≥ Phase 0-4 的 bar(容差 ±0.5 mAP)。
  - 7 個共同 class 的 per-class AP 無單類崩掉。註:native 的 truck 吸收了 construction_vehicle 等
    類別,truck AP 偏高屬預期,不可反向解讀為「整體更好」。

## Phase 2 — T4 base DB(35 ep)

- [ ] **2-1. 訓練**(init 直接吃 Phase 1 checkpoint,零轉檔、零 drop-pattern):
      ```bash
      autoware-ml train \
          --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
          --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt \
          datamodule.data_root=<t4_data_root>
      ```
- [ ] **2-2. sanity**:val mAP 走勢對照轉檔路線的 stage-2 經驗值(有 AWML base-stage 數字就一併記這裡)。

## Phase 3 — j6gen2 finetune + 最終驗收

- [ ] **3-1. 訓練**:
      ```bash
      autoware-ml train \
          --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune \
          --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt \
          datamodule.data_root=<t4_data_root>
      ```
- [ ] **3-2. 最終驗收(真正的 acceptance;Phase 1 只是中繼)**:aligned eval(±51.2 m 方形 GT、
      min-points filter、0-121 m bucket)對比轉檔路線既有數字:

      | 路線 | val mAP | test mAP |
      |---|---|---|
      | 轉檔路線(autoware-ml, run 92068f7b) | 0.39127 | 0.36609 |
      | AWML aligned_bf16 | 0.37521 | 0.35515 |
      | **native 畢業路線(本計畫)** | (待填) | (待填) |

      **±1 mAP 內 → 宣告畢業**。顯著偏低 → 回 Phase 1 歸因(優先查 AMP、sampler、camera order)。

## Phase 4 — 收尾

- [ ] **4-1. docs/models/streampetr.md 改寫**:native 三階段升為主線;
      model-zoo 轉檔流程降級為「快速起手的替代路徑」;convert tool 定位改為 backbone init 一次性步驟。
- [ ] **4-2. Deploy/ONNX 驗證**:RGB-native checkpoint 匯出 + 輸出比對一次
      (runtime 端 color order 問題從此不存在,順手在 deploy docs 註記輸入是 RGB)。
- [ ] **4-3. 本文件歸檔**:結果數字填完後,精華段落(差異表 + 最終數字)併入 streampetr.md,本文件刪除。

## Non-goals

- **重訓 FCOS3D/DD3D backbone**:成本遠超收益。畢業後唯一保留的外部 artifact 就是這顆
  (且已翻成 RGB、native 命名)。明確不做。
- 10-class nuScenes parity 重現:7-class 是 autoware-ml 的有意設計,不回頭改。
- TensorRT engine 驗證(現況 ONNX 為止,維持不變)。

## 成本估算

| 項目 | 估算 |
|---|---|
| Phase 0 | ~半天(upstream 下載 + remap 檢查 + 兩項驗證) |
| Phase 1 | 2 GPU × ~4 天(AWML 基準:4×H100 × ~2 天) |
| Phase 2 | 35 ep T4 base DB(比照既有 stage-2 經驗) |
| Phase 3 | 35 ep j6gen2(比照既有 finetune 經驗) |

## 狀態記錄

| 日期 | 事項 |
|---|---|
| 2026-08-10 | 計畫建立。前提驗證完成(RGB-native pipeline、配方齊備、FCOS3D artifact 下落待查)。 |
| 2026-08-10 | Phase 0-1 改為 upstream StreamPETR 公開 artifact + 自行 remap 為主路徑;AWML 內部檔案降級為 optional 交叉驗證。 |
| 2026-08-10 | **Phase 0-1、0-2 實測完成**(autoware-ml-yihsiang container):upstream 檔即 remapped 版,零 remap;convert + 等價性 + harness 載入全過。產物:`work_dirs/ckpts/fcos3d_vovnet_imgbackbone_rgb_native.pth`。 |
| 2026-08-10 | **0-3 完成 + Phase 1 訓練流程端到端冒煙通過**(新 container `autoware-ml-yihsiang-nusc`,nuScenes 掛載 `/mnt/qnapdata/external/nuscenes`):① 單卡 `fast_dev_run` exit 0;② **2-GPU DDP** 限量 run(3 train + 2 val batches)exit 0 — 權重雙 rank 載入、validation `val/loss` 記錄、checkpoint 寫入 mlruns。踩坑兩個已記錄:`--device /dev/nvidia-uvm`、短測需 `check_val_every_n_epoch=1`。**剩 0-4(acceptance bar)後即可正式開跑 Phase 1。** |
| 2026-08-10 | **三個 stage 全部冒煙通過**(container 重建為雙掛載:internal→/workspace/data、external nuscenes→/workspace/datasets/nuscenes):stage 1(nuScenes pretrain)、stage 2(T4 base,`info/detection3d` infos)、stage 3(j6gen2 finetune,`info/kokseang_2_8` infos,parity 同款 split)fast_dev_run 全 exit 0。**stage 1 checkpoint → stage 2/3 載入 1526/1526 全覆蓋**,跨 stage 零轉檔實證。正式訓練將在 DGX Station A100 執行 → 完整指令見 `STREAMPETR_DGX_RUNBOOK.md`(4×A100 用 `batch_size=4 trainer.devices=4` 保總 batch 16、LR 不動;GPU 3 是 DGX Display 要排除)。 |
