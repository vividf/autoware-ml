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

"""Stage-graph declaration, the generic pipeline, and artifact discovery — on toy stages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autoware_ml.deployment.export import available_backends
from autoware_ml.deployment.pipeline import PipelineCache, StagedPipeline
from autoware_ml.deployment.stages import (
    GraphStage,
    TorchStage,
    artifact_path,
    final_stage,
    graph_stages,
    validate_stages,
)
from autoware_ml.types.backend import Backend


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


def _toy_stages():
    return (
        TorchStage("prep", run=lambda ctx: {"x": ctx.batch_inputs.x.float()}),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage("glue", run=lambda ctx: {"z": ctx["y"].to(ctx.device) + 0.5}),
        GraphStage(
            "head",
            module=_SplitHead(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"), ("minus", "b")),
        ),
    )


class TestStageDeclaration:
    def test_validate_returns_tuple_and_final_stage(self):
        stages = validate_stages(_toy_stages())
        assert [s.name for s in graph_stages(stages)] == ["encoder", "head"]
        assert final_stage(stages).name == "head"

    def test_duplicate_names_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("encoder", run=lambda ctx: {})
        with pytest.raises(ValueError, match="Duplicate"):
            validate_stages(stages)

    def test_final_stage_needs_output_fields(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage(
            "head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")
        )
        with pytest.raises(ValueError, match="output_fields"):
            validate_stages(stages)

    def test_only_final_stage_may_declare_output_fields(self):
        stages = list(_toy_stages())
        stages[1] = GraphStage(
            "encoder", module=_Double(), inputs=("x",), outputs=("y",), output_fields=(("y", "f"),)
        )
        with pytest.raises(ValueError, match="Only the final"):
            validate_stages(stages)

    def test_output_fields_must_name_declared_outputs(self):
        with pytest.raises(ValueError, match="not among its outputs"):
            GraphStage(
                "h", module=_Double(), inputs=("x",), outputs=("y",), output_fields=(("q", "f"),)
            )

    def test_no_graph_stage_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_stages([TorchStage("only", run=lambda ctx: {})])

    def test_artifact_paths_derive_from_stage_name(self, tmp_path):
        assert artifact_path(tmp_path, "encoder", Backend.ONNX) == tmp_path / "encoder.onnx"
        assert artifact_path(tmp_path, "encoder", "tensorrt") == tmp_path / "encoder.engine"
        with pytest.raises(ValueError):
            artifact_path(tmp_path, "encoder", Backend.PYTORCH)


class TestStagedPipeline:
    def test_pytorch_backend_runs_all_stages_in_order(self):
        pipeline = StagedPipeline(_toy_stages(), backend="pytorch", device=torch.device("cpu"))
        batch = SimpleNamespace(x=torch.tensor([1, 2, 3]))
        result, context = pipeline.run(batch)

        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        assert torch.equal(result.outputs["plus"], torch.tensor([3.5, 5.5, 7.5]))
        assert torch.equal(result.outputs["minus"], torch.tensor([1.5, 3.5, 5.5]))
        assert result.output_names == ["plus", "minus"]
        assert result.graph_stage_names == ("encoder", "head")
        assert set(result.stage_times_ms) == {"prep", "encoder", "glue", "head"}
        assert set(context.tensors) == {"x", "y", "z", "plus", "minus"}

    def test_assemble_maps_onnx_names_to_fields(self):
        pipeline = StagedPipeline(
            _toy_stages(),
            backend=Backend.PYTORCH,
            device=torch.device("cpu"),
            assemble=lambda fields: dict(fields),
        )
        result = pipeline.infer(SimpleNamespace(x=torch.tensor([1.0])))
        assembled = pipeline.assemble(result)
        assert set(assembled) == {"a", "b"}
        assert assembled["a"].item() == pytest.approx(3.5)

    def test_assemble_without_hook_raises(self):
        pipeline = StagedPipeline(_toy_stages(), backend="pytorch", device=torch.device("cpu"))
        result = pipeline.infer(SimpleNamespace(x=torch.tensor([1.0])))
        with pytest.raises(RuntimeError, match="assemble_predictions"):
            pipeline.assemble(result)

    def test_non_pytorch_backend_requires_artifacts_dir(self):
        with pytest.raises(ValueError, match="artifacts_dir"):
            StagedPipeline(_toy_stages(), backend="onnx", device=torch.device("cpu"))

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            StagedPipeline(_toy_stages(), backend="tflite", device=torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage(
            "head",
            module=_Double(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"),),
        )
        pipeline = StagedPipeline(stages, backend="pytorch", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="returned 1 tensor"):
            pipeline.infer(SimpleNamespace(x=torch.tensor([1.0])))

    def test_cache_builds_each_backend_device_once(self, tmp_path):
        cache = PipelineCache(_toy_stages(), tmp_path, assemble=lambda f: f)
        first = cache.get("pytorch", "cpu")
        assert cache.get(Backend.PYTORCH, torch.device("cpu")) is first


class TestAvailableBackends:
    def _touch(self, path, mtime=None):
        import os

        path.write_bytes(b"stub")
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def test_pytorch_always_available(self, tmp_path):
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH}

    def test_onnx_requires_every_graph_stage_file(self, tmp_path):
        self._touch(tmp_path / "encoder.onnx")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH}
        self._touch(tmp_path / "head.onnx")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH, Backend.ONNX}

    def test_tensorrt_requires_every_graph_stage_file(self, tmp_path):
        self._touch(tmp_path / "encoder.engine")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH}
        self._touch(tmp_path / "head.engine")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH, Backend.TENSORRT}

    def test_stale_engine_stays_available_but_warns(self, tmp_path, caplog):
        import logging

        for name in ("encoder", "head"):
            self._touch(tmp_path / f"{name}.engine", mtime=1_000)
            self._touch(tmp_path / f"{name}.onnx", mtime=2_000)
        with caplog.at_level(logging.WARNING):
            available = available_backends(_toy_stages(), tmp_path)
        assert Backend.TENSORRT in available
        assert "STALE TENSORRT ENGINE" in caplog.text


def test_graph_stage_declares_its_own_dynamic_axes_and_the_config_overrides_them(
    tmp_path, monkeypatch
) -> None:
    """Axes intrinsic to a graph live on the stage; ``deploy.stages`` still wins.

    A point model has no static point count, so its axes are a property of the
    declaration rather than a per-experiment choice.
    """
    from omegaconf import OmegaConf
    import torch
    from torch import nn

    from autoware_ml.deployment import export as export_module
    from autoware_ml.deployment.config import DeployConfig
    from autoware_ml.deployment.stages import GraphStage, TorchStage

    seen: list[dict] = []

    def fake_export_to_onnx(module, args, path, **kwargs):
        seen.append(kwargs["dynamic_axes"])
        path.write_bytes(b"")

    monkeypatch.setattr(export_module, "export_to_onnx", fake_export_to_onnx)

    declared = {"x": {0: "num_points"}, "y": {0: "num_points"}}
    stage = GraphStage(
        "points",
        module=nn.Identity(),
        inputs=("x",),
        outputs=("y",),
        output_fields=(("y", "y"),),
        onnx_dynamic_axes=declared,
    )

    def seed(context):
        return {"x": torch.ones(1, 2)}

    def run(stages_config: dict) -> dict:
        seen.clear()
        deploy_cfg = DeployConfig.from_dict(
            OmegaConf.create(
                {
                    "onnx": {"enabled": True, "dynamo": False, "opset_version": 17},
                    "tensorrt": {"enabled": False},
                    "stages": stages_config,
                }
            )
        )
        export_module.export_stages(
            (TorchStage("seed", run=seed), stage),
            batch_inputs=None,
            deploy_cfg=deploy_cfg,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
        return seen[0]

    assert run({}) == declared
    configured = {"x": {0: "batch"}}
    assert run({"points": {"onnx": {"dynamic_axes": configured}}}) == configured


def test_export_honors_the_per_stage_precision_override(tmp_path, monkeypatch) -> None:
    """`deploy.stages.<name>.onnx.precision` wins over the global fp16 setting."""
    from omegaconf import OmegaConf
    import torch
    from torch import nn

    from autoware_ml.deployment import export as export_module
    from autoware_ml.deployment.config import DeployConfig
    from autoware_ml.deployment.stages import GraphStage, TorchStage

    monkeypatch.setattr(
        export_module, "export_to_onnx", lambda *args, path=None, **kwargs: args[2].write_bytes(b"")
    )
    converted: list[str] = []
    monkeypatch.setattr(
        export_module, "autocast_to_fp16", lambda path, inputs: converted.append(path.stem)
    )
    monkeypatch.setattr(export_module, "onnx_has_qdq", lambda path: False)
    monkeypatch.setattr(export_module, "onnx_custom_op_domains", lambda path: ())

    def seed(context):
        return {"x": torch.ones(1, 2)}

    stages = (
        TorchStage("seed", run=seed),
        GraphStage("kept_fp32", module=nn.Identity(), inputs=("x",), outputs=("mid",)),
        GraphStage(
            "goes_fp16",
            module=nn.Identity(),
            inputs=("mid",),
            outputs=("y",),
            output_fields=(("y", "y"),),
        ),
    )
    deploy_cfg = DeployConfig.from_dict(
        OmegaConf.create(
            {
                "onnx": {"enabled": True, "dynamo": False, "opset_version": 17, "precision": "fp16"},
                "tensorrt": {"enabled": False},
                "stages": {"kept_fp32": {"onnx": {"precision": "fp32"}}},
            }
        )
    )
    export_module.export_stages(
        stages, batch_inputs=None, deploy_cfg=deploy_cfg, output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert converted == ["goes_fp16"]
