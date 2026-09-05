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

"""Architecture recipes: residual-block conversion (shared vs fresh residual quantizer,
class matching, numerics with quantizers off), the VoVNet OSA / eSE placements, and the
MaxPool wrapper. VoVNet is exercised through stand-in modules that carry the attribute
structure the quantized forwards read (the real classes live outside this repo)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule  # noqa: E402
from torch import nn  # noqa: E402

from autoware_ml.quantization.config import Precision, QuantizationConfig  # noqa: E402
from autoware_ml.quantization.core.quantizer_state import quantizers_disabled  # noqa: E402
from autoware_ml.quantization.core.replace import replace_quantizable_modules  # noqa: E402
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules  # noqa: E402
from autoware_ml.quantization.recipes.attach import (  # noqa: E402
    RECIPE_ATTACHERS,
    BlockSpecs,
    ESEBlockSpec,
    RecipeContext,
    ResidualBlockSpec,
    attach_ese_recipe,
    attach_residual_add_recipe,
)
from autoware_ml.quantization.recipes.quant_blocks import (  # noqa: E402
    QuantBeforePool,
    QuantESEModule,
    QuantOSAModule,
)


class _DenseBlock(nn.Module):
    """A ResNet-style basic block (test-local; the repo's own residual block is spconv-based)."""

    def __init__(self, channels: int, downsample: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv2d(channels, channels, 1) if downsample else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv2(self.relu(self.conv1(x))) + identity)


class _QuantDenseBlock(QuantModule):
    def _setup(self) -> None:
        pass

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        residual_quantizer = getattr(self, "residual_quantizer", None)
        if residual_quantizer is not None:
            identity = residual_quantizer(identity)
        return self.relu(self.conv2(self.relu(self.conv1(x))) + identity)


_SPEC = ResidualBlockSpec(
    _DenseBlock, _QuantDenseBlock, share_from=("conv1",), fresh_if_downsample=True
)


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _DenseBlock(4)
        self.down = _DenseBlock(4, downsample=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.down(self.block(x)))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = _Body()

    def forward(self, x):
        return self.body(x)


def _ctx(roots=("body",), skip=frozenset(), specs=None, on_apply=None) -> RecipeContext:
    return RecipeContext(
        precision=Precision.INT8,
        calibrator="histogram",
        roots=roots,
        skip_names=frozenset(skip),
        specs=specs if specs is not None else BlockSpecs(residual=(_SPEC,)),
        on_apply=on_apply,
    )


def _apply(recipe: str, model: nn.Module, skip=frozenset(), on_apply=None) -> int:
    return RECIPE_ATTACHERS[recipe](model, _ctx(skip=skip, on_apply=on_apply))


class TestResidualAddRecipe:
    def test_matched_blocks_convert_in_place_with_shared_or_fresh_quantizer(self):
        model = _Model().eval()
        replace_quantizable_modules(
            model.body, kinds=("conv",), prefix="body", precision=Precision.INT8
        )
        block, down = model.body.block, model.body.down
        record = []
        count = attach_residual_add_recipe(
            model, _ctx(roots=("body",), on_apply=lambda *a: record.append(a))
        )
        assert count == 2
        # In place, class patched, identity kept.
        assert model.body.block is block and isinstance(block, _DenseBlock)
        assert type(block).__name__ == "Quant_DenseBlock"
        # Plain block shares conv1's input quantizer (plain attribute: not in the state_dict).
        assert block.residual_quantizer is block.conv1.input_quantizer
        assert "residual_quantizer" not in "".join(block.state_dict().keys())
        # Downsample block gets its own (a submodule, so its amax will be saved).
        assert isinstance(down.residual_quantizer, TensorQuantizer)
        assert down.residual_quantizer is not down.conv1.input_quantizer
        assert "residual_quantizer" in down._modules
        transforms = {(name, transform) for name, transform, _, _ in record}
        assert transforms == {("body.block", "convert_block"), ("body.down", "convert_block")}
        details = {name: detail for name, _, _, detail in record}
        assert "shared from conv1.input_quantizer" in details["body.block"]
        assert "fresh (block has a downsample branch)" in details["body.down"]

    def test_quantized_forward_equals_original_with_quantizers_off(self):
        torch.manual_seed(0)
        model = _Model().eval()
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            want = model(x)
        replace_quantizable_modules(
            model.body, kinds=("conv",), prefix="body", precision=Precision.INT8
        )
        _apply("residual_add", model)
        with torch.no_grad(), quantizers_disabled(model):
            got = model(x)
        assert torch.equal(got, want)

    def test_unmatched_and_already_converted_blocks_are_left_alone(self):
        model = _Model().eval()
        assert _apply("residual_add", model) == 2
        assert _apply("residual_add", model) == 0  # idempotent

        class _Other(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 2, 1)

        other = _Other()
        assert attach_residual_add_recipe(other, _ctx(roots=("conv1",))) == 0

    def test_subclass_with_its_own_forward_does_not_match(self):
        class _Custom(_DenseBlock):
            def forward(self, x):
                return x

        model = nn.Module()
        model.custom = _Custom(4)
        with pytest.raises(KeyError):
            attach_residual_add_recipe(model, _ctx(roots=("custom",)))

    def test_rules_declare_model_owned_blocks(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "fuse_bn": False, "ptq": {"calibrate_samples": 1}}
        )
        rules = QuantRules(quantize_submodules={"body": ("conv",)}, residual_blocks=(_SPEC,))
        model = _Model().eval()
        plan = QuantizationPlan(rules=rules, config=config)
        plan.prepare(model)
        by_transform = {}
        for d in plan.placement_record.decisions:
            by_transform.setdefault(d.transform, []).append(d.module)
        assert sorted(by_transform["convert_block"]) == ["body.block", "body.down"]
        assert by_transform["wrap_module"] == ["body.pool"]
        assert sorted(by_transform["replace_module"]) == [
            "body.block.conv1",
            "body.block.conv2",
            "body.down.conv1",
            "body.down.conv2",
            "body.down.downsample",
        ]


class TestRecipeScope:
    def test_recipes_fire_only_inside_the_quantized_submodules(self):
        """A residual block / pool outside ``quantize_submodules`` gets no Q/DQ (BEVFusion's
        spconv encoder holds SparseBasicBlocks but deploys through libspconv)."""

        class _Two(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quantized = _Body()
                self.plain = _Body()

        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "fuse_bn": False, "ptq": {"calibrate_samples": 1}}
        )
        rules = QuantRules(quantize_submodules={"quantized": ("conv",)}, residual_blocks=(_SPEC,))
        model = _Two().eval()
        plan = QuantizationPlan(rules=rules, config=config)
        plan.prepare(model)
        touched = sorted(d.module for d in plan.placement_record.decisions)
        assert all(name.startswith("quantized.") for name in touched)
        assert type(model.plain.block) is _DenseBlock
        assert isinstance(model.plain.pool, nn.MaxPool2d) and not isinstance(
            model.plain.pool, QuantBeforePool
        )
        assert isinstance(model.quantized.pool, QuantBeforePool)


class TestMaxPoolRecipe:
    def test_wraps_pools_and_honors_skip_subtrees(self):
        model = _Model().eval()
        record = []
        assert _apply("maxpool", model, on_apply=lambda *a: record.append(a)) == 1
        assert isinstance(model.body.pool, QuantBeforePool)
        assert isinstance(model.body.pool.pool, nn.MaxPool2d)
        assert record[0][:2] == ("body.pool", "wrap_module")
        assert _apply("maxpool", model) == 0  # idempotent

        skipped = _Model().eval()
        assert _apply("maxpool", skipped, skip={"body"}) == 0
        assert isinstance(skipped.body.pool, nn.MaxPool2d)

    def test_wrapper_is_the_identity_with_the_quantizer_off(self):
        torch.manual_seed(0)
        model = _Model().eval()
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            want = model(x)
        _apply("maxpool", model)
        with torch.no_grad(), quantizers_disabled(model):
            assert torch.equal(model(x), want)


@pytest.mark.skipif(
    pytest.importorskip("importlib").util.find_spec("spconv") is None, reason="spconv not installed"
)
class TestSparseBasicBlockSpec:
    def test_default_spec_converts_the_repo_block(self):
        from autoware_ml.models.detection3d.encoders.sparse import SparseBasicBlock
        from autoware_ml.quantization.recipes.attach import default_block_specs

        specs = default_block_specs()
        assert [s.block_cls for s in specs.residual] == [SparseBasicBlock]
        assert specs.ese == ()
        model = nn.Module()
        model.block = SparseBasicBlock(4, "k0", 1e-3, 0.01)
        replace_quantizable_modules(
            model, kinds=("conv",), precision=Precision.INT8
        )  # spconv convs are not nn.Conv2d
        count = attach_residual_add_recipe(model, _ctx(roots=("block",), specs=specs))
        assert count == 1
        assert type(model.block).__name__ == "QuantSparseBasicBlock"
        # No shareable input quantizer on a SubMConv3d -> fresh quantizer, saved in the state_dict.
        assert "block.residual_quantizer" in " ".join(model.state_dict().keys()) or isinstance(
            model.block.residual_quantizer, TensorQuantizer
        )


# --------------------------------------------------------------------------- VoVNet
# Stand-ins with the attribute structure of VoVNet's _OSA_module / eSEModule (the real
# classes live in the model package that owns them).


class _ESELike(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.hsigmoid(self.fc(self.avg_pool(x)))


class _OSALike(nn.Module):
    def __init__(self, channels: int = 4, layers: int = 3, identity: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
                for _ in range(layers)
            ]
        )
        self.concat = nn.Sequential(nn.Conv2d(channels * (layers + 1), channels, 1), nn.ReLU())
        self.ese = _ESELike(channels)
        self.identity = identity
        self.depthwise = False
        self.isReduced = False

    def forward(self, x):  # mirrors VoVNet _OSA_module.forward
        identity_feat = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        xt = self.ese(self.concat(torch.cat(output, dim=1)))
        return xt + identity_feat if self.identity else xt


class _Counting(nn.Module):
    """Identity stand-in for a TensorQuantizer that counts how many tensors passed through."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x


_VOV_SPECS = BlockSpecs(
    residual=(
        ResidualBlockSpec(_OSALike, QuantOSAModule, share_from=("concat.0",), osa_concat=True),
    ),
    ese=(ESEBlockSpec(_ESELike, QuantESEModule),),
)


class _VoVModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.osa = _OSALike(identity=True)
        self.backbone.osa_noid = _OSALike(identity=False)

    def forward(self, x):
        return self.backbone.osa_noid(self.backbone.osa(x))


def _prepare_vov(model: nn.Module, disable=()) -> list:
    config = QuantizationConfig.from_dict(
        {
            "enabled": True,
            "mode": "ptq",
            "fuse_bn": False,
            "disable_recipes": list(disable),
            "ptq": {"calibrate_samples": 1},
        }
    )
    rules = QuantRules(
        quantize_submodules={"backbone": ("conv", "linear")},
        residual_blocks=_VOV_SPECS.residual,
        ese_blocks=_VOV_SPECS.ese,
    )
    plan = QuantizationPlan(rules=rules, config=config)
    plan.prepare(model)
    return plan.placement_record.decisions


class TestVoVNetRecipes:
    def test_placement_structure(self):
        model = _VoVModel().eval()
        decisions = _prepare_vov(model)
        converted = {d.module: d.detail for d in decisions if d.transform == "convert_block"}
        assert set(converted) == {
            "backbone.osa",
            "backbone.osa_noid",
            "backbone.osa.ese",
            "backbone.osa_noid.ese",
        }
        osa = model.backbone.osa
        assert type(osa).__name__ == "Quant_OSALike" and isinstance(osa, _OSALike)
        assert len(osa.concat_input_quantizers) == 3 and not hasattr(osa, "residual_quantizer")
        assert "single Q at the block input" in converted["backbone.osa"]
        assert "identity=False" in converted["backbone.osa_noid"]
        ese = osa.ese
        assert type(ese).__name__ == "Quant_ESELike"
        assert isinstance(ese.pool_input_quantizer, TensorQuantizer)
        assert isinstance(ese.mul_gate_quantizer, TensorQuantizer)
        # Fresh quantizers are submodules (state_dict keys once calibrated).
        assert "pool_input_quantizer" in ese._modules and "concat_input_quantizers" in osa._modules

    def test_numerics_equal_original_with_quantizers_off(self):
        torch.manual_seed(0)
        model = _VoVModel().eval()
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            want = model(x)
        _prepare_vov(model)
        with torch.no_grad(), quantizers_disabled(model):
            got = model(x)
        assert torch.allclose(got, want, atol=1e-6)

    def test_single_q_fan_out(self):
        """OSA (identity=True): block input quantized ONCE and reused for layer 0, Concat and
        the Add; eSE: input quantized ONCE and reused for the gate path and the Mul bypass."""
        model = _VoVModel().eval()
        _prepare_vov(model)
        osa = model.backbone.osa
        spies = nn.ModuleList([_Counting() for _ in osa.concat_input_quantizers])
        osa.concat_input_quantizers = spies
        osa.ese.pool_input_quantizer = _Counting()
        osa.ese.mul_gate_quantizer = _Counting()
        with torch.no_grad(), quantizers_disabled(model):
            osa(torch.randn(1, 4, 8, 8))
        assert [spy.calls for spy in spies] == [
            1,
            1,
            1,
        ]  # input + layer0 + layer1 outputs; last layer not
        assert osa.ese.pool_input_quantizer.calls == 1 and osa.ese.mul_gate_quantizer.calls == 1

    def test_disable_ese_recipe_leaves_ese_plain(self):
        model = _VoVModel().eval()
        decisions = _prepare_vov(model, disable=("ese",))
        converted = [d.module for d in decisions if d.transform == "convert_block"]
        assert converted and all(not m.endswith(".ese") for m in converted)
        assert type(model.backbone.osa.ese) is _ESELike

    def test_ese_recipe_is_scoped_and_idempotent(self):
        model = nn.Module()
        model.head = _ESELike(4)
        model.other = _ESELike(4)
        ctx = _ctx(roots=("head",), specs=_VOV_SPECS)
        assert attach_ese_recipe(model, ctx) == 1
        assert attach_ese_recipe(model, ctx) == 0
        assert type(model.other) is _ESELike
