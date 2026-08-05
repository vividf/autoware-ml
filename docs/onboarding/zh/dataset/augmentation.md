# Augmentation 與 Transform

> **本文涵蓋內容：** `BaseTransform` 合約、`TransformsCompose`、transform 函式庫的
> 版面配置，以及如何閱讀／撰寫一個 transform。Transform 是 CPU、逐 sample 的階段，
> 負責**載入檔案並對資料做 augment**。
>
> 先備知識：[dataset_pipeline.md](dataset_pipeline.md)。

---

## 1. 為什麼會有 transform（以及為什麼載入邏輯放在這裡）

`Dataset` 只會回傳*metadata*（路徑、原始標註）。其他所有事情 — 讀取 LiDAR
檔案、堆疊 sweep、把標註轉換成框（box）、翻轉／旋轉／縮放、依範圍裁切 — 都是由
**transform** 完成的。原因有兩個：

1. **可組合性（Composability）。** Pipeline 是你在 config 中組出的一份有序清單。
   可以依實驗替換 augmentation，而不需要動到 Python 程式碼。
2. **依 split 而異的行為。** Train 會套用隨機 augmentation；val/test/predict 則只做
   確定性（deterministic）的載入＋裁切。同一個 dataset，不同的 pipeline。

Transform 是**在 CPU 上、於 DataLoader 的 worker process 中、以每個 sample 為單位**
執行的。繁重的 GPU、以 batch 為單位的工作（voxelize）*不是* transform — 那是模型自有
的 `DataPreprocessing`（見 [dataset_pipeline.md](dataset_pipeline.md)）。

---

## 2. 合約：dict-in / dict-out（`transforms/base.py:28`）

每個 transform 都是一個 `BaseTransform`。它會從 sample dict 中讀取一些 key，並且
**只回傳它變更過的 key**；由 composer 負責把這些結果合併回去。

```python
class BaseTransform(ABC):
    p: float | None = None          # apply probability (None = always)
    _required_keys: Sequence[str] = ()   # KeyError if any missing
    _optional_keys: Sequence[str] = ()   # triggers apply_defaults() if missing
    pre_transform: Any = None

    def __call__(self, input_dict, context=None):        # :48
        self._context = context
        self._validate_required_keys(input_dict)         # 1. required keys → KeyError if absent
        self._handle_optional_keys(input_dict)           # 2. fill optional defaults
        if not self._should_apply():                     # 3. probability gate
            return self.on_skip(input_dict)              #    (default: return unchanged)
        return self.transform(input_dict)                # 4. the real work

    @abstractmethod
    def transform(self, input_dict) -> dict: ...         # :153  YOU implement — return UPDATES
```

這四個固定步驟（`__call__:48`）意味著撰寫 transform 的人只需要寫 `transform()`。
基底類別會統一處理驗證、optional key 的預設值，以及機率閘門（probability gate）。

`_should_apply()`（`:112`）：`p is None` → 一定套用；`p<=0` → 一定不套用；`p>=1` →
一定套用；其餘情況則是 `np.random.rand() < p`。所以隨機性的 augmentation 要設定
`p`，loader 與確定性操作則保持 `None`。

### 組合（Composition）（`transforms/base.py:167`）

```python
class TransformsCompose:
    def __init__(self, pipeline=()):
        self.pipeline = list(pipeline)

    def __call__(self, input_dict, context=None):        # :182
        for transform in self.pipeline:
            output = transform(input_dict, context=context)
            if not isinstance(output, dict): raise TypeError(...)   # each transform MUST return a dict
            input_dict |= output                         # :203  merge updates
        return input_dict
```

`input_dict |= output` 這個合併動作就是整個合約的核心：一個 transform 回傳它有
動到的 key 子集合，這些 key 會覆寫目前執行中的 dict。這就是為什麼一個 loader 可以
只回傳 `{"points": ...}`，而一個 augmentation 可以只回傳
`{"points": ..., "gt_boxes": ...}`。

---

## 3. 閱讀真實的 transform

### 一個 loader — `LoadPointsFromFile`（`transforms/point_cloud/loading.py:27`）

```python
class LoadPointsFromFile(BaseTransform):
    _required_keys = ["lidar_path"]          # will KeyError without it

    def __init__(self, *, load_dim=5, use_dim=(0, 1, 2, 3)):
        self.load_dim, self.use_dim = load_dim, use_dim

    def transform(self, input_dict):
        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)
        # optional single-source slicing / sensor-frame transform ...
        points = points[:, list(self.use_dim)] if not isinstance(self.use_dim, int) else points[:, :self.use_dim]
        return {"points": points.astype(np.float32)}    # only the key it produced
```

重點：`_required_keys` 宣告了它的輸入合約；它從 metadata 中讀取一個*路徑*，並回傳
一個 `points` array；`p` 是 `None`（loader 一定會執行）。

### 一個 augmentation — `GlobalRotScaleTrans`（`transforms/point_cloud/geometry.py:112`）

```python
class GlobalRotScaleTrans(BaseTransform):
    _required_keys = []

    def __init__(self, *, rot_range, scale_ratio_range, translation_std=None): ...

    def transform(self, input_dict):
        g3d.require_point_cloud(input_dict)
        rotation, rotation_angle, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std)
        g3d.transform_points(input_dict, rotation, scale, translation)   # points
        g3d.transform_normal(input_dict, rotation)                       # normals (if present)
        g3d.transform_boxes(input_dict, rotation, rotation_angle, scale, translation)  # gt_boxes
        return input_dict
```

**關鍵細節：** 一個 augmentation 會**同時**變換 **points 與框（以及 normal）**。
`RandomFlip3D`（`geometry.py:75`）也是一樣 — 每一次 flip 都會套用到 `points`、
`normal`，*以及* `gt_boxes`。如果你寫的 augmentation 只移動了 points 卻忘了處理框，
你的標籤就會在不知不覺中失去同步，訓練也會悄悄地變差。所有相關數學運算都集中在
`autoware_ml/transforms/geometry3d.py`（`g3d`）之中，而 camera／camera-lidar 的
變體也重用完全相同的函式，讓純 LiDAR、camera 與 fusion 的 augmentation 保持一致。

---

## 4. Transform 函式庫地圖（`autoware_ml/transforms/`）

請把 `_target_` 指向具體的實作模組（沒有透過 `__init__` 重新匯出）。

| 資料夾 | 用途 | 代表性 transforms |
| ------ | ------- | ------------------------- |
| `common/` | 與模態（modality）無關 | `Copy`、`BuildPointFeatures`、`PermuteAxes` |
| `point_cloud/` | LiDAR | **loading：** `LoadPointsFromFile`、`LoadPointsFromMultiSweeps`；**geometry：** `GlobalRotScaleTrans`、`RandomFlip3D`、`RandomRotateTargetAngle`；**crop：** `PointsRangeFilter`、`CropBoxInner/Outer`、`SphereCrop`；**sampling：** `PointShuffle`、`RandomDropout`、`ElasticDistortion`、`GridSample`；**perturbation：** `RandomJitter`、`RandomShift`；**formatting：** `PreparePointCloudInput` |
| `boxes3d/` | 3D 標註 | `LoadAnnotations3D`（→ `gt_boxes`、`gt_names`、`gt_labels`、`gt_num_points`）、`MergeObjects3D`、filter 類 `ObjectRangeFilter`/`ObjectNameFilter`/`ObjectMinPointsFilter`/`ObjectRangeMinPointsFilter` |
| `camera/` | 影像 | `LoadImageFromFile`、`LoadMultiViewImagesFromFiles`、resize／crop／flip、normalize、`GridMask`、`UndistortImage` |
| `camera_lidar/` | fusion | `LidarCameraFusion`、`CalibrationMisalignment`、`ImageAug3D`、`BEVLoadMultiViewImageFromFiles` |
| `image/` | 僅限 2D | `PhotometricDistortion` |
| `segmentation3d/` | seg 標籤／augmentation | `LoadSegAnnotations3D`、`PreparePointSegInput`、`FrustumMix`、`InstanceCopy`、`RangeInterpolation` |
| `multi_task/` | 有型別的 multi-task 堆疊 | `MultiTaskTransformsCompose` ＋自有的 loading／geometry（操作 Pydantic sample） |

---

## 5. 依 split 設定 pipeline

Transform 是以 `TransformsCompose` 搭配 `pipeline:` 清單的方式接起來的。以下取自
CenterPoint 的 NuScenes leaf config：

```yaml
datamodule:
  train_transforms:
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.boxes3d.loading.LoadAnnotations3D
        name_mapping: ${dataset.detection3d.name_mapping}
      - _target_: autoware_ml.transforms.point_cloud.sweeps.LoadPointsFromMultiSweeps
        sweeps_num: 10
        load_dim: 5
        use_dim: [0, 1, 2, 3, 4]
      - _target_: autoware_ml.transforms.point_cloud.geometry.RandomFlip3D
        flip_ratio_bev_horizontal: 0.5
        flip_ratio_bev_vertical: 0.5
      - _target_: autoware_ml.transforms.point_cloud.geometry.GlobalRotScaleTrans
        rot_range: [-1.571, 1.571]
        scale_ratio_range: [0.9, 1.1]
        translation_std: [0.5, 0.5, 0.2]
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.boxes3d.filters.ObjectRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.point_cloud.sampling.PointShuffle

  val_transforms:                       # ← NO random augmentation
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.boxes3d.loading.LoadAnnotations3D
        name_mapping: ${dataset.detection3d.name_mapping}
      - _target_: autoware_ml.transforms.point_cloud.sweeps.LoadPointsFromMultiSweeps
        sweeps_num: 10
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}

  test_transforms: ${datamodule.val_transforms}    # reuse val's pipeline verbatim
```

從這裡可以看出的經驗法則：

- **順序很重要。** 先載入標註與 points；再做 augment；然後 crop／filter；最後才
  shuffle。
- **Val/test/predict 不可以做隨機 augmentation。** 只能做確定性的載入＋範圍裁切。
  `test_transforms: ${datamodule.val_transforms}` 可以保證兩者一致。
- **`predict_transforms` 通常會省略標註載入**（推論時沒有 label）。

---

## 6. 撰寫新的 transform（作法）

```python
# autoware_ml/transforms/point_cloud/my_aug.py
from typing import Any
import numpy as np
from autoware_ml.transforms.base import BaseTransform

class RandomIntensityScale(BaseTransform):
    """Scale the per-point intensity channel. Reads/writes `points`."""
    _required_keys = ["points"]          # fail loudly if points aren't loaded yet

    def __init__(self, *, p: float = 0.5, scale_range=(0.9, 1.1)):
        self.p = p                        # BaseTransform's gate handles the probability
        self.scale_range = scale_range

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict["points"]
        scale = np.random.uniform(*self.scale_range)
        points[:, 3] = points[:, 3] * scale     # column 3 = intensity
        return {"points": points}               # return only what you changed
```

接著把它加進 config 中的 `pipeline:` 清單。檢查清單：

- 宣告 `_required_keys`，讓順序錯誤的 pipeline 能丟出清楚的 `KeyError`。
- **只**回傳有變更過的 key。
- 隨機行為要設定 `p`；確定性操作保持 `None`。
- 如果這個 augmentation 會移動幾何資訊，要**同時變換 points 與框**（重用 `g3d`
  輔助函式）。
- 保持在 CPU／numpy 的範圍內（它是在 worker 中執行）。GPU 的工作屬於
  `preprocessing/`。

---

## 7. Transform 特有的注意事項

- **原地（in-place）變更。** 許多 transform 會原地變更 array，*同時*也回傳它們。
  這沒問題，因為每個 sample 在各個 worker 中是彼此獨立的 — 但不要跨 sample
  快取／共用 array。
- **就算產生了某個 key，還是可能被丟棄**，如果它不在 `collation_map` 中。在
  transform 中產生 `gt_names`，並不代表它就會傳到模型，除非 collation 有保留它。
- **用於混合的 `context`。** Copy-paste 風格的 augmentation（`FrustumMix`、
  `InstanceCopy`）使用 `self.context.sample_secondary(...)` 來取得另一個 sample。
  不要直接伸手進 dataset 裡拿。
- **如果你宣告了 `_optional_keys`，就必須實作 `apply_defaults`**，否則基底類別
  會丟出 `NotImplementedError`（`base.py:135`）。

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| `KeyError: Missing required key 'points'` | transform 排在它的 loader 之前 | 把 loader 排在 pipeline 最前面 |
| 開啟 augmentation 後框偏移／mAP 崩潰 | 某個 augmentation 移動了 points 卻沒有動到框 | 把 points **與** `gt_boxes` 一起變換（使用 `g3d`） |
| `TypeError: ... must return a dict` | `transform()` 回傳了 `None`／array | 回傳一個變更內容的 dict |
| 評估（eval）期間 augmentation 仍在作用 | `val/test_transforms` 中有隨機 transform | 保持 val/test 為確定性；透過 `${...}` 重用 val 的設定 |
| 執行結果不可重現 | RNG 沒有設定 seed | 訓練透過 `L.seed_everything(..., workers=True)` 設定 seed；各 transform 使用 `np.random`（由 worker 設 seed） |
| 修改 transform 卻沒有效果 | 改到的 `_target_` 跟 config 指向的不是同一個 | 確認 pipeline 中確切的模組路徑 |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 新增一個 augmentation | 撰寫一個 `BaseTransform`，加進 `train_transforms.pipeline` |
| 調整 augmentation 的強度 | 編輯 config 中該 transform 的參數（`rot_range`、`scale_ratio_range`、`p` 等） |
| 為了做 ablation 而關閉 augmentation | 從 `train_transforms.pipeline` 移除該項目（或設定 `p: 0.0`） |
| 載入額外的特徵通道 | 使用 `LoadPointsFromFile`/`LoadPointsFromMultiSweeps` 的 `load_dim`/`use_dim`，再把該通道貫穿整個模型 |
| 新增一個混合型 augmentation | 使用 `self.context.sample_secondary(...)`；可參考 `FrustumMix`/`InstanceCopy` 的做法 |

---

**Next (Phase 3):** [../model/model_architecture.md](../model/model_architecture.md) — 模型
如何消費（consume）這個 batch，並將其轉換成 prediction 與 loss。
