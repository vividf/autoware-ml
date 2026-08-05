# DataLoader OOM(SIGKILL `-9`)—— StreamPETR 多卡訓練記憶體爆掉的分析與驗證

> 情境:在共享機 `dpc2011001` 上,以 2 GPU 跑 StreamPETR
> (`vov_480x640_t4dataset_j6gen2_base_2gpu`),`num_workers=32` 時訓練在
> **Epoch 3 中途被系統強制殺掉**。本文記錄:症狀、根因、判斷依據、量測驗證、
> 把 `num_workers` 降到 8 之後**如何驗證問題不會再發生**,以及事後發現的
> **孤兒 worker 殘留(OOM 之後 61 個 dataloader worker 沒被回收,壓了 40 GB
> 在 swap 裡)的診斷與安全清理程序**(§9)。

---

## 1. 症狀(原始 log)

```text
Epoch 3/9 ━━━━━━╺━━ 2314/3351 ...
INFO: [rank: 1] Child process with PID 9487 terminated with code -9.
      Forcefully terminating all other processes to avoid zombies 🧟
```

拆解這行:

| 片段 | 意義 |
| --- | --- |
| `rank: 1` | 2 張卡裡的第 2 個 DDP 訓練 process(每卡一個 process) |
| `terminated with code -9` | 被 **SIGKILL(訊號 9)** 殺掉 —— 程式沒有自己丟例外,是**作業系統核心**動手 |
| `Forcefully terminating all other processes` | Lightning 偵測到 rank 1 死了,把 rank 0 也一起收掉,避免殭屍 process |

**`-9` 不是 CUDA OOM。** GPU 記憶體不足會是 `RuntimeError: CUDA out of memory`。
核心主動送 SIGKILL,在 Linux 上幾乎只有一個常見來源:**系統 RAM 不足,OOM
killer 砍掉記憶體用最兇的 process。**

---

## 2. 結論(TL;DR)

- 不是程式 bug,是 **系統 RAM(不是 GPU 記憶體)被 dataloader worker 撐爆**。
- 根因:autoware-ml 的 dataset 把整份 annotation 存成**純 Python `list[dict]`**
  (train ≈ **5 GB** 活物件),在 `fork` 出來的多個 worker 下,因 **CPython 的
  reference-count copy-on-write** 效應被逐一私有化複製,RAM 隨 worker 數線性放大。
- `num_workers=32` × 2 卡 = **64 個 worker** → 私有化總量遠超 125 GB → OOM。
- **不是 worker 數本身的問題**:AWML(mmengine)同樣 `num_workers=32` 不會死,
  因為 mmengine 預設 `serialize_data=True`,把 annotation 打包成**共享 bytes
  buffer**,64 個 worker 共用一份。詳見 §5。
- **緩解**:`num_workers=8`(2 卡共 16 worker)把放大倍率降到 1/4,可安全跑完。
  `num_workers` 只是 loader 效能旋鈕,**不影響訓練出來的權重**。
- **治本(已實作)**:`SerializedSampleList` 把 annotation 收進 fork 共享的
  序列化 buffer,實測每 worker 私有化從 230 MB 降到 7.4 MB(31×),已套用到
  全部 8 個 dataset、`num_workers` 恢復 32。見 §7.2。
- **注意殘留**:被 SIGKILL 的那輪會留下**孤兒 dataloader worker**(本次 61 個、
  壓了 40 GB 在 swap),不清掉就重跑會更容易再 OOM。診斷與安全清理見 §9。

---

## 3. 根本原因與程式碼佐證

### 3.1 資料被存成純 Python list

[`autoware_ml/datamodule/common/multiview_detection3d.py`](../../autoware_ml/datamodule/common/multiview_detection3d.py) 的 dataset `__init__`:

```python
with open(ann_file, "rb") as file:
    data = pickle.load(file)
self.data_infos = load_detection_data_infos(data)   # ← 純 list[dict]
```

[`autoware_ml/datamodule/common/detection3d.py`](../../autoware_ml/datamodule/common/detection3d.py) 的
`load_detection_data_infos`:

```python
def load_detection_data_infos(data):
    return [normalize_detection_sample(sample) for sample in data["data_list"]]
```

`self.data_infos` 是一個 **list,裡面每個 element 是一個巢狀 dict**(每個 frame 的
相機/光達路徑、標定、3D 框…)。沒有任何序列化 / 共享記憶體處理。

### 3.2 為什麼多 worker 會讓它爆掉 —— copy-on-write refcount

DataLoader `num_workers>0` 時,PyTorch 用 **`fork`** 開子行程(本機實測預設就是
`fork`)。`fork` 之後父子行程的記憶體是 **copy-on-write(CoW)** 共享的:理論上只要
不寫入,大家共用同一份 `data_infos`,RAM 不會放大。

問題出在 **CPython 的物件模型**:每個 Python 物件開頭都有一個
**reference count**,而**光是「讀取 / 走訪」一個物件就會改動它的 refcount**
(`Py_INCREF`/`Py_DECREF`)。refcount 就存在物件自己那塊記憶體頁裡 → 一讀就寫 →
**那一頁的 CoW 觸發,被複製成該 worker 私有**。

於是每個 worker 跑一個 epoch、走訪過大部分 frame 之後,就會把 `data_infos` 的
大半頁面各自複製一份。**64 個 worker = 最多接近 64 份 5 GB 的私有副本**,RAM 隨
worker 數近似線性成長。這是 PyTorch 社群有名的
「**DataLoader + 大型 in-memory dataset 記憶體膨脹**」現象。

### 3.3 為什麼是「Epoch 3 才死」而不是一開始

- `persistent_workers` 預設 **False**([`autoware_ml/datamodule/base.py`](../../autoware_ml/datamodule/base.py) `persistent_workers: bool = False`),
  所以 worker 每個 epoch 重生一次 → 記憶體用量是**每個 epoch 一個鋸齒**,在
  epoch 尾端(worker 走訪過最多資料時)達到峰值。
- 峰值本來就已經逼近上限;共享機上又有別人約 39 GB 在用、swap 也已吃掉 34 GB。
  前兩個 epoch 靠 swap 硬撐過去,到 **Epoch 3 的峰值 + 累積壓力**就把可用記憶體
  推到 0,核心於是 SIGKILL 掉 rank 1。這解釋了「撐一陣子才掛」的模式。

### 3.4 `persistent_workers` 跟這個問題的關係

`persistent_workers` 是這個 CoW 問題的**放大器開關**:它決定 worker 活多久,
而 worker 活多久決定它有多少時間把共享頁「摸髒」。

**`False`(目前兩邊的設定)—— 每個 epoch 重生,私有化有上限。**
worker 在 epoch 開始時 fork(此刻與主行程 100% 共享),epoch 內只會摸到
分給它的那 **1/N 筆樣本**所在的頁面,epoch 結束整批砍掉、記憶體歸還 →
就是 §3.3 的**鋸齒型**:每個 epoch 峰值大致相同、不跨 epoch 累積。

**`True` —— worker 永生,私有化一路累積。**
sampler 每個 epoch 重新洗牌,同一個 worker 這個 epoch 摸這批、下個 epoch 摸
另一批 → 幾個 epoch 後**每個 worker 都逐漸摸過整份資料** → 私有化趨近
§4.3 表格的「最壞(100%)」欄,而且中間永遠不歸還。

| 設定 | 記憶體行為 | 意義 |
| --- | --- | --- |
| `False`(現況) | 鋸齒、峰值有界 | 已是傷害最小的設定;但 32 workers 連「有界的峰值」都超標,照樣爆 |
| `True` | 單調累積到最壞值 | 會**更早、更嚴重**地爆 |

**常見誤解澄清:「把它設成 False 能不能改進?」—— 不能,它本來就是 False**:
autoware-ml 預設值在 [`autoware_ml/datamodule/base.py`](../../autoware_ml/datamodule/base.py)
(`persistent_workers: bool = False`),AWML 的 config 也明寫 `False`。
本次 OOM 是「False 之下、單一 epoch 內的峰值」就已超標,root cause 仍是
list[dict] 表示法 × worker 數(§3.1–3.2),解法不變:降 `num_workers`(治標)
或 serialize `data_infos`(治本,§7.2)。

`False` 的代價:每個 epoch 開頭要重新 fork 全部 worker、重建 prefetch,
所以每個 epoch 開始會卡一小段。目前值得付;等 §7.2 的 serialize 做完、
資料頁不再會被摸髒,`persistent_workers=True` + 高 worker 數才會變成
安全的加速選項。

---

## 4. 我怎麼判斷 / 怎麼量測驗證的

判斷不是靠猜,是靠下面幾個可重現的檢查:

### 4.1 確認機器記憶體狀態(崩潰後快照)

```console
$ free -h
              total   used   free  shared  buff/cache  available
Mem:          125Gi   39Gi   7.4Gi  44Gi    78Gi        40Gi
Swap:          92Gi   34Gi   57Gi
$ df -h /dev/shm      # 63G,只用 8.5G → 不是 shm 不夠
$ nproc               # 48 → 64 個 worker 已超過核心數
```

要點:**swap 已用 34 GB**、`shared` 高達 44 GB(worker 間共享/私有頁),
而 `/dev/shm` 沒滿 → 排除「共享記憶體不足」,鎖定「系統 RAM 不足」。
(`dmesg` 需 root 才看得到 oom-kill 那行,本機無權限;但 `-9` + 上述狀態已足夠定性。)

### 4.2 量測 annotation 在記憶體裡到底多大

用遞迴 `sys.getsizeof` 深度量測(見 §6 可重現腳本),結果:

| pkl | 磁碟大小 | frames | 反序列化後 Python 物件記憶體 |
| --- | --- | --- | --- |
| `..._val.pkl` | 50 MB | 3,215 | **270 MB**(5.4× disk,~84 KB/frame) |
| `..._train.pkl` | 941 MB | 56,573 | **≈ 4.7–5.1 GB**(56,573 × 84 KB) |

反序列化成 Python 物件會**膨脹約 5 倍**(dict/list/str 的物件標頭開銷)。
train 的 `data_infos` 活物件約 **5 GB**。

### 4.3 用數字對帳:64 vs 16 worker

以 5 GB 基準、CoW 私有化比例估算單卡+雙卡總量(worst = 每個 worker 把整份都摸過):

| 設定 | worker 數 | 私有化 30% | 私有化 50% | 最壞(100%) |
| --- | --- | --- | --- | --- |
| `num_workers=32`(舊) | 64 | ~96 GB | ~160 GB | ~320 GB |
| `num_workers=8`(新) | 16 | ~24 GB | ~40 GB | ~77 GB |

- 125 GB 機器、又有 ~39 GB 被別人佔用 → 可用其實只有 ~85 GB。
  64 worker 的 30–50% 私有化(96–160 GB)**必然超標** → 與「Epoch 3 OOM」一致。
- 16 worker 的 30–50%(24–40 GB)**落在安全範圍**,加上 base ~5 GB 與別人用量仍有餘裕。

數字方向完全對得上崩潰行為,這是「降 worker 數就能解」的量化依據。

---

## 5. 為什麼 AWML(mmengine)同樣 `num_workers=32` 不會死

**不是 worker 數不同,是資料存法不同。**

| | AWML(mmengine) | autoware-ml(Lightning) |
| --- | --- | --- |
| annotation 存法 | 序列化成**共享 bytes buffer** | 純 Python `list[dict]` |
| 依據 | `BaseDataset` 預設 **`serialize_data=True`**:把 data_list 打包成一整塊 `np.uint8` buffer + 位址表 | 無序列化 |
| 64 workers 下 | 全部共用**同一份** buffer → RAM 幾乎常數 | 各自 CoW 私有化 → RAM 線性成長 |
| `num_workers=32` 結果 | 沒事 | Epoch 3 OOM |

mmengine 就是為了避開 §3.2 的 CoW refcount 問題,才預設把資料序列化成一塊
「沒有 Python 物件標頭、也就沒有 refcount 可寫」的連續 bytes,worker 只在需要時
反序列化單筆。所以它開 64 個 worker 也不會爆。

### 5.1 常見疑問:「autoware-ml 不是也用同一種 info pkl 嗎?」

對 —— **兩邊讀的是同一種(甚至同一份)info pkl,pkl 本身不是問題**。
差別在「讀進來之後,資料以什麼形式常駐記憶體」。

讀檔那一步兩邊完全一樣:

```text
AWML:        pickle.load(ann_file)  →  data_list  = list[dict]
autoware-ml: pickle.load(ann_file)  →  data_infos = list[dict]
```

**分岔在下一步:**

**autoware-ml —— 就停在這裡。**
[`multiview_detection3d.py`](../../autoware_ml/datamodule/common/multiview_detection3d.py)
把 `list[dict]` 直接存成 `self.data_infos` 永久持有。56,573 個 frame =
**幾百萬個 Python 小物件**(dict/str/list/float),每個物件頭上都有 refcount;
worker fork 後只要「讀」到任何一筆,refcount 就被寫入 → 該頁 CoW 複製成私有。

**AWML(mmengine)—— 多做一步「序列化再收起來」。**
`BaseDataset.full_init()` 讀完 pkl 後呼叫 `_serialize_data()`:

```python
# mmengine/dataset/base_dataset.py(serialize_data=True 是預設)
self.data_bytes, self.data_address = self._serialize_data()
# 每筆 dict → pickle 成 bytes,全部塞進一整塊 np.uint8 陣列 + 一個 offset 表,
# 然後把原本的 list[dict] 丟掉
```

之後 `__getitem__` 要用哪一筆,才臨時 `pickle.loads(data_bytes[start:end])`
解出**那一筆**,用完即丟。

**為什麼這一步決定生死**:`data_bytes` 是「一個」Python 物件,941 MB 的內容是
純 bytes,**沒有任何 Python 物件標頭、也就沒有 refcount 可被寫**:

| | autoware-ml `list[dict]` | mmengine `data_bytes` |
| --- | --- | --- |
| Python 物件數 | 幾百萬個 | **2 個**(bytes 陣列 + offset 表) |
| worker 讀資料時 | 走訪物件 → 寫 refcount → CoW 觸發 | 只做 slice,底層頁面**從不被寫** |
| fork 後 64 workers | 各自私有化 ~5 GB → 爆 | **全部共用同一份 ~1 GB** |

結論:「用 info pkl」這件事兩邊相同;錯在 autoware-ml 把 pkl 解開後,以
「多物件、會被 refcount 寫髒」的形式常駐。mmengine 特地把它壓回「單一塊
不可變 bytes」,就是為了讓 fork 出來的 worker 能真正共享 —— 這也正是 §7.2
治本方案要照抄的模式。

---

## 6. 可重現的量測腳本(§4.2 用的)

```python
# measure_datainfos_mem.py
import pickle, sys, os, gc

def deep_size(obj, seen=None):
    if seen is None: seen = set()
    if id(obj) in seen: return 0
    seen.add(id(obj))
    s = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items(): s += deep_size(k, seen) + deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for x in obj: s += deep_size(x, seen)
    return s

base = "/mnt/qnapdata/internal/t4datasets/info/kokseang_2_8/"
for name in ["t4dataset_j6gen2_base_infos_val.pkl",
             "t4dataset_j6gen2_base_infos_train.pkl"]:
    p = base + name
    d = pickle.load(open(p, "rb"))
    n = len(d["data_list"])
    print(f"{name}: disk={os.path.getsize(p)/1e6:.0f}MB frames={n}")
    if "val" in name:  # train 太大,只量 val 再外推
        mem = deep_size(d["data_list"]) / 1e6
        print(f"   in-memory ~= {mem:.0f}MB  (~{mem/n*1000:.1f}KB/frame)")
    del d; gc.collect()
```

---

## 7. 修法

### 7.1 過渡期緩解:`num_workers=8`(已由 §7.2 取代)

治本落地前的止血措施:把
[`vov_480x640_t4dataset_j6gen2_base_2gpu.yaml`](../../autoware_ml/configs/tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu.yaml)
的 `num_workers` 從 32 降到 8(2 卡共 16 worker,放大倍率 1/4),讓訓練先能
跑完。`num_workers` 只影響資料載入速度,**不改變訓練結果**,所以不破壞跟
AWML 的 recipe 對齊。§7.2 的序列化實作合入後,此值已恢復 32。

### 7.2 治本:`SerializedSampleList`(已實作,2026-07-24)

已仿 mmengine `serialize_data` 實作
[`autoware_ml/datamodule/common/serialization.py`](../../autoware_ml/datamodule/common/serialization.py)
的 **`SerializedSampleList`**:初始化時把每筆 sample `pickle.dumps` 進一塊連續
buffer + int64 offset 表,`__getitem__` 才 `pickle.loads` 單筆。介面上是
`len()` + 整數索引的 drop-in 替代,回傳的是**全新反序列化副本**(對取出樣本的
修改不會寫回)。

**套用範圍(8 個 dataset,全部在 `__init__` 尾端、所有全量走訪之後收攏)**:

| dataset | 額外處理 |
| --- | --- |
| `common/multiview_detection3d.py` | `scene_index_groups()` 改為 init 時預先快取(sampler 在 fork 後呼叫,不能再全量反序列化) |
| `t4dataset/detection3d.py` | `frame_weights` 先在活 list 上算完再序列化 |
| `nuscenes/detection3d.py` | — |
| `t4dataset/segmentation3d.py` | — |
| `nuscenes/segmentation3d.py` | — |
| `t4dataset/segdet.py` | 同 `frame_weights` |
| `nuscenes/segdet.py` | 同 `frame_weights` |
| `t4dataset/calibration_status.py` | — |

**驗證**:

- 單元測試 `autoware_ml/tests/datamodule/test_serialization.py`(roundtrip、
  邊界、迭代、fork 子行程共享讀取、可 pickle);datamodule 全套 54 個 +
  全 repo 542 個測試通過(排除本來就未編譯 `bev_pool_ext` 的 2 個收集錯誤)。
- **CoW 私有化實測**(真實 val pkl,3,215 frames,活物件 270 MB;fork 4 個
  child,各深度走訪全部樣本兩輪 + GC,量 `/proc/self/smaps_rollup` 的
  Private_Dirty):

| 存法 | 每 worker 私有化 | 佔資料集比例 |
| --- | --- | --- |
| `list[dict]`(舊) | **230 MB** | ~85% —— 幾乎整份複製,印證 §3.2 |
| `SerializedSampleList`(新) | **7.4 MB** | 固定 allocator 開銷,**不隨資料集大小成長** |

**31× 降幅**。換算 train pkl(活物件 ~5 GB):舊法 64 workers ≈ 275 GB
(正是本次 OOM 的量級);新法與 worker 數、資料集大小近乎無關。

量測時的一個坑(供重現參考):序列化模式的父行程若還握著原始活 list 再 fork,
child 的 `gc.collect()` 會把**繼承來的那份**也摸髒,量出來假性偏高 —— 量測前
父行程必須先 `del` 活 list(真實 dataset `__init__` 結束後本來就沒有活 list)。

實作後 [`vov_480x640_t4dataset_j6gen2_base_2gpu.yaml`](../../autoware_ml/configs/tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu.yaml)
的 `num_workers` 已恢復為 AWML parity 的 32。

---

## 8. 如何驗證 `num_workers=8` 不會再爆(給你現在做)

因為 `persistent_workers=False`,記憶體是**每個 epoch 一個鋸齒**,峰值出現在每個
epoch 尾端且**大致重複**。所以驗證邏輯很簡單:

> **只要 Epoch 1–2 的記憶體峰值離上限有明顯餘裕,後面每個 epoch 也一樣安全** ——
> 之前正是因為峰值已頂到上限,才在 Epoch 3 被壓力波動壓垮。

### 步驟 A:開一個背景記憶體監測(在 host 上,另開一個終端)

```bash
LOG=~/mem_trace_$(date +%F_%H%M).log
while true; do
  ts=$(date +%T)
  read used avail <<<"$(free -m | awk '/Mem:/{print $3, $7}')"
  swap=$(free -m | awk '/Swap:/{print $3}')
  echo "$ts mem_used=${used}MB avail=${avail}MB swap_used=${swap}MB" | tee -a "$LOG"
  sleep 15
done
```

(記憶體是 host 層級的 —— 容器用 `--ipc=host`,所以在 host 上 `free` 看到的就是
訓練的真實用量。)

### 步驟 B:重跑訓練(`num_workers=8` 已在 config 內)

```bash
autoware-ml train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu \
  datamodule.data_root=/workspace/data/t4datasets \
  datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl \
  --weights /workspace/pretrained/nuscenes_vov99_baseline_320x800_converted.pth
```

### 步驟 C:判讀 —— 通過條件

盯 `mem_trace` log 與訓練進度,滿足以下即可判定安全:

- [ ] **撐過 Epoch 3 step 2314**(上次死掉的那一點),並順利進入 Epoch 4。
- [ ] `avail` 在每個 epoch 峰值時仍 **> ~10–15 GB**,不會歸零。
- [ ] `swap_used` **不是單調一路往上爬**到接近 92 GB(鋸齒/持平 OK,持續逼頂危險)。
- [ ] 沒有再出現 `terminated with code -9`。

判讀重點:

| 現象 | 意義 |
| --- | --- |
| 每個 epoch 峰值**大致持平**、離上限有餘裕 | ✅ 安全,可放心跑完 10 epoch |
| 峰值**逐 epoch 墊高**、avail 越來越少 | ⚠️ 仍有累積,降到 `num_workers=4` 再測 |
| avail 觸 0 / swap 逼頂 / 再度 `-9` | ❌ 還是不夠,降 worker 或先避開共享機尖峰時段 |

### 步驟 D:想更省時間(可選)

不用等真的訓練 3 小時 —— 只要 dataloader 開始餵資料、跑進第 1 個 epoch 一段時間
(例如前 10–20 分鐘)觀察峰值趨勢即可。因為峰值機制每 epoch 重複,第 1 個 epoch
的峰值就有代表性。

---

## 9. 後續事件:OOM 之後的孤兒 worker 殘留與清理(2026-07-24 實錄)

### 9.1 現象

重跑 `num_workers=8` 的訓練後,機器記憶體仍異常吃緊(available 只剩 ~19 GB、
swap 用到 53 GB)。懷疑上一輪(`num_workers=32`,被 OOM 殺掉那輪)有殘留。

盤點後確認:**rank 主行程被 SIGKILL 時,它 fork 出來的 dataloader worker 不會
被自動回收** —— SIGKILL 不給程式做清理的機會,worker 變成孤兒(reparent 到
容器 init),繼續佔著記憶體:

| 批次 | 啟動時間 | 狀態 | process 數 | 記憶體 |
| --- | --- | --- | --- | --- |
| 舊輪 `num_workers=32` | 00:41 | 11:09 主行程被 OOM 殺 | **61 個孤兒 worker 存活 ~14 小時** | RSS ~5 GB + **~40 GB 被擠進 swap** |
| 新輪 `num_workers=8` | 12:00 | 正常訓練中 | 34 個 | GPU 每卡 ~27 GB |

這批殘留和新輪一起擠,讓機器比實際更緊 —— 也就是說 **OOM 一次之後如果不清理,
下一輪的可用記憶體會更少,更容易再 OOM**(惡性循環)。

### 9.2 怎麼診斷(唯讀,不影響訓練)

核心技巧:**用「啟動時間(lstart)」把新舊兩輪的 process 分開**,因為兩輪的
指令列長得一模一樣,只有啟動時間能區分。

```bash
# 1. 列出所有訓練相關 process,帶啟動時間 —— 看有幾個「批次」
ps -eo pid,ppid,user,stat,lstart,rss,etimes,args --sort=lstart \
  | grep -iE "scripts.train|autoware-ml train|pt_data_worker" | grep -v grep

# 2. 按啟動時間分組計數 —— 一輪訓練的所有 process 啟動時間會擠在同一分鐘內
ps -eo lstart,args | grep -iE "scripts.train|autoware-ml train" | grep -v grep \
  | awk '{print $1,$2,$3,$4}' | sort | uniq -c

# 3. 分批統計 process 數與 RSS(把 00:41 換成舊輪的啟動時間)
ps -eo pid,lstart,rss,args | grep -iE "scripts.train|pt_data_worker" | grep -v grep \
  | awk '/ 00:41:/{a++; m+=$8} END{printf "舊輪: %d procs, RSS %.1f GB\n", a, m/1024/1024}'

# 4. GPU 端確認(哪些 PID 佔 GPU 記憶體、是否還存在)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

判讀要點:

- 舊輪的 rank 主行程已不在,但同啟動時間的 `pt_data_worker` 一大批還在
  (`stat=Sl`,etimes 十幾小時)→ **孤兒 worker 殘留**。
- RSS 加總看起來不大(5 GB)但 swap 很高 → 殘留頁面大多被換出到 swap,
  `free` 的 `used` 看不出來,**要看 swap 用量**。
- `nvidia-smi --query-compute-apps` 可能列出 `[Not Found]` 的殭屍 PID;
  以 `memory.used` 總量為準 —— 若每卡用量 = 現役訓練的量,GPU 就是乾淨的,
  那些只是驅動的殘留紀錄,不佔實際 VRAM。

### 9.3 怎麼安全清理(現役訓練還在跑的情況下)

**絕對不要 `pkill -f python` / `pkill -f train`** —— 新舊兩輪指令列相同,
會把正在跑的訓練一起殺掉。安全做法是**逐 PID 三重驗證後才 kill**:

```bash
# PID 清單來自 §9.2 步驟 1,只收「舊輪啟動時間」的那批
for pid in <舊輪的PID們>; do
  # 三重保險:PID 還存在 + 啟動時間吻合舊輪 + 指令確實是這個訓練
  if ps -o lstart=,args= -p $pid 2>/dev/null | grep " 00:41:" | grep -q "streampetr"; then
    kill -9 $pid
  fi
done
```

三重驗證的理由:

1. **PID 存在檢查**:worker 可能已自己退場。
2. **啟動時間吻合**:Linux PID 會回收重用 —— kill 前一刻再驗一次啟動時間,
   杜絕「PID 已被新 process 接手」的誤殺。
3. **指令列吻合**:確保不是碰巧同 PID 的無關程式。

### 9.4 清理結果(驗證)

```text
killed=61 skipped=0
00:41 舊輪剩餘:0
12:00 現役輪:34(不變,訓練未受影響)
```

| 指標 | 清理前 | 清理後 |
| --- | --- | --- |
| swap used | 53 GB | **13 GB(釋放 40 GB)** |
| RAM available | 19 GB | 24 GB |
| GPU(每卡 98 GB) | — | 各 ~27 GB,恰為現役訓練的兩個 rank → 乾淨 |

### 9.5 教訓:訓練被 `-9` 殺掉之後,重跑前先做這件事

```bash
# 有沒有上一輪的孤兒 worker?(啟動時間明顯早於你要重跑的時刻就是殘留)
ps -eo pid,lstart,etimes,args | grep -iE "scripts.train|pt_data_worker" | grep -v grep
free -h   # swap 異常高也是殘留的訊號
```

有殘留就照 §9.3 清掉再重跑,否則新一輪是在「被殘留吃掉一大塊」的機器上跑,
更容易再次 OOM。

---

## 附:關鍵檔案

- Dataset(存 `data_infos` 的地方):
  [`autoware_ml/datamodule/common/multiview_detection3d.py`](../../autoware_ml/datamodule/common/multiview_detection3d.py)
- `load_detection_data_infos`:
  [`autoware_ml/datamodule/common/detection3d.py`](../../autoware_ml/datamodule/common/detection3d.py)
- `persistent_workers` / dataloader 預設:
  [`autoware_ml/datamodule/base.py`](../../autoware_ml/datamodule/base.py)
- 2-GPU 訓練 config(已設 `num_workers=8`):
  [`autoware_ml/configs/tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu.yaml`](../../autoware_ml/configs/tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu.yaml)
