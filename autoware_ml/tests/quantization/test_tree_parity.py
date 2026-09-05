# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-plan-same-tree parity tests for CenterPoint quantization.

The invariant every quantization stage relies on: the PTQ producer, the QAT
callback, and the deploy loader all build the quantized module tree by calling
the model's one ``build_quantization_plan`` — so two independently prepared
models must have byte-identical state_dict key sets, and a producer state_dict
must load into a loader-prepared tree with ``strict=True``.

Requires nvidia-modelopt (skipped otherwise).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone  # noqa: E402
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import (  # noqa: E402
    PillarFeatureNet,
)
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import (  # noqa: E402
    PointPillarsScatter,
)
from autoware_ml.models.detection3d.heads.centerhead import CenterHead  # noqa: E402
from autoware_ml.models.detection3d.main_modules.centerpoint import (  # noqa: E402
    CenterPointDetectionModel,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN  # noqa: E402
from autoware_ml.models.multi_task_base_model import LogDictConfigs  # noqa: E402
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor  # noqa: E402
from autoware_ml.quantization.config import QuantizationConfig  # noqa: E402

_POINT_CLOUD_RANGE = [-10.0, -10.0, -3.0, 10.0, 10.0, 5.0]
_VOXEL_SIZE = [0.5, 0.5, 8.0]


def _tiny_centerpoint() -> CenterPointDetectionModel:
    torch.manual_seed(0)
    voxel_encoder = PillarFeatureNet(
        in_channels=5,
        feat_channels=[8, 8],
        voxel_size=_VOXEL_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
    )
    middle_encoder = PointPillarsScatter(in_channels=8, output_shape=[40, 40])
    backbone = SECONDBackbone(
        in_channels=8,
        out_channels=[16, 32],
        layer_nums=[1, 1],
        layer_strides=[1, 2],
        activation_checkpointing=False,
    )
    neck = SECONDFPN(
        in_channels=[16, 32],
        out_channels=[16, 16],
        upsample_strides=[1, 2],
        activation_checkpointing=False,
    )
    head = CenterHead(
        in_channels=32,
        class_names=["car", "pedestrian"],
        shared_channels=8,
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        out_size_factor=2,
        min_radius=2,
        score_threshold=0.0,
        post_max_size=10,
        nms_min_radius=1.0,
        use_velocity=True,
    )
    return CenterPointDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        pts_voxel_encoder=voxel_encoder,
        pts_middle_encoder=middle_encoder,
        pts_backbone=backbone,
        pts_neck=neck,
        bbox_head=head,
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )


_CONFIG = QuantizationConfig.from_dict(
    {
        "enabled": True,
        "mode": "ptq",
        "skip_quantize": ["pts_voxel_encoder"],
        "disable_recipes": ["residual_add"],
        "ptq": {"calibrate_samples": 4},
    }
)


class TestTreeParity:
    def test_same_plan_builds_identical_key_sets(self):
        model_a = _tiny_centerpoint()
        model_b = _tiny_centerpoint()
        model_a.build_quantization_plan(_CONFIG).prepare(model_a)
        model_b.build_quantization_plan(_CONFIG).prepare(model_b)
        assert set(model_a.state_dict().keys()) == set(model_b.state_dict().keys())

    def test_two_prepares_record_identical_placements(self):
        model_a = _tiny_centerpoint()
        model_b = _tiny_centerpoint()
        plan_a = model_a.build_quantization_plan(_CONFIG)
        plan_b = model_b.build_quantization_plan(_CONFIG)
        plan_a.prepare(model_a)
        plan_b.prepare(model_b)
        only_a, only_b = plan_a.placement_record.diff(plan_b.placement_record)
        assert only_a == [] and only_b == []
        assert len(plan_a.placement_record) > 0

    def test_producer_state_dict_loads_strict_into_loader_tree(self):
        producer = _tiny_centerpoint()
        producer.build_quantization_plan(_CONFIG).prepare(producer)
        state_dict = producer.state_dict()

        loader = _tiny_centerpoint()
        loader.build_quantization_plan(_CONFIG).prepare(loader)
        incompatible = loader.load_state_dict(state_dict, strict=True)
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys

    def test_fuse_bn_removes_batchnorm_and_adds_bias(self):
        model = _tiny_centerpoint()
        keys_before = set(model.state_dict().keys())
        model.build_quantization_plan(_CONFIG).prepare(model)
        keys_after = set(model.state_dict().keys())
        # BN statistics disappear from the fused towers...
        assert any("running_mean" in key for key in keys_before)
        assert not any(
            "running_mean" in key for key in keys_after if key.startswith("pts_backbone")
        )
        # ...and the fused convolutions gain a bias.
        assert any(
            key.startswith("pts_backbone") and key.endswith("conv.bias") for key in keys_after
        )

    def test_skip_quantize_subtree_has_no_quantizers(self):
        from modelopt.torch.quantization.nn import TensorQuantizer as tq_cls

        model = _tiny_centerpoint()
        model.build_quantization_plan(_CONFIG).prepare(model)
        kept = [
            name
            for name, module in model.named_modules()
            if isinstance(module, tq_cls) and name.startswith("pts_voxel_encoder")
        ]
        assert kept == []
        quantized = [
            name
            for name, module in model.named_modules()
            if isinstance(module, tq_cls) and name.startswith("pts_backbone")
        ]
        assert quantized, "the backbone tower must carry quantizers"
        # modelopt naming: the calibrated scales live under input_quantizer / weight_quantizer.
        assert all(
            name.endswith(("input_quantizer", "weight_quantizer", "output_quantizer"))
            for name in quantized
        )
