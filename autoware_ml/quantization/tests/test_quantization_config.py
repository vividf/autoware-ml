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

"""Tests for the typed quantization config (typo guard, mode lies, defaults)."""

from __future__ import annotations

import pytest

from autoware_ml.quantization.config import (
    CalibrationConfig,
    Precision,
    PTQConfig,
    QATConfig,
    QATScheduleConfig,
    QuantizationConfig,
)


class TestQuantizationConfig:
    def test_absent_section_is_disabled(self):
        config = QuantizationConfig.from_dict(None)
        assert not config.enabled
        assert config.mode == "ptq"
        assert config.fuse_bn

    def test_round_trip(self):
        config = QuantizationConfig.from_dict(
            {
                "enabled": True,
                "mode": "ptq",
                "skip_quantize": ["pts_voxel_encoder"],
                "disable_recipes": ["residual_add"],
                "ptq": {"calibrate_samples": 400, "batch_size": 1, "calib_seed": 0},
            }
        )
        assert config.enabled
        assert config.skip_quantize == ("pts_voxel_encoder",)
        assert config.disable_recipes == ("residual_add",)
        assert config.ptq == PTQConfig(calibrate_samples=400, batch_size=1, calib_seed=0)

    def test_typo_guard_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="skip_quantizes"):
            QuantizationConfig.from_dict({"enabled": True, "skip_quantizes": ["x"]})

    def test_qat_block_under_ptq_mode_is_a_config_lie(self):
        with pytest.raises(ValueError, match="config lie"):
            QuantizationConfig.from_dict(
                {"enabled": True, "mode": "ptq", "qat": {"epochs": 1, "lr": 1e-4}}
            )

    def test_ptq_block_under_qat_mode_is_a_config_lie(self):
        with pytest.raises(ValueError, match="config lie"):
            QuantizationConfig.from_dict(
                {"enabled": True, "mode": "qat", "ptq": {"calibrate_samples": 400}}
            )

    def test_explicit_null_producer_block_is_fine(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "qat", "ptq": None, "qat": {"epochs": 3, "lr": 1e-4}}
        )
        assert config.ptq is None
        assert config.qat == QATConfig(epochs=3, lr=1e-4)

    def test_default_precision_values(self):
        config = QuantizationConfig.from_dict({"enabled": True, "default_precision": "fp8"})
        assert config.default_precision is Precision.FP8
        # Unknown precisions still die at parse time.
        with pytest.raises(ValueError, match="valid values"):
            QuantizationConfig.from_dict({"enabled": True, "default_precision": "int4"})
        config = QuantizationConfig.from_dict({"enabled": True, "default_precision": "int8"})
        assert config.default_precision is Precision.INT8
        assert QuantizationConfig.from_dict(config.to_dict()) == config


class TestCalibrationConfig:
    def test_default_is_histogram_mse(self):
        config = QuantizationConfig.from_dict({"enabled": True})
        assert config.calibration == CalibrationConfig()
        assert config.calibration.method == "mse"
        assert config.calibration.activation_calibrator == "histogram"
        assert config.to_dict()["calibration"] == {"method": "mse"}

    def test_accepts_method_string_and_mapping(self):
        assert CalibrationConfig.from_raw("entropy").method == "entropy"
        config = CalibrationConfig.from_raw({"method": "percentile", "percentile": 99.9})
        assert config.percentile == 99.9
        assert config.to_dict() == {"method": "percentile", "percentile": 99.9}
        assert CalibrationConfig.from_raw(config.to_dict()) == config

    def test_max_and_smoothquant_use_the_max_calibrator(self):
        assert CalibrationConfig.from_raw("max").activation_calibrator == "max"
        config = CalibrationConfig.from_raw({"method": "smoothquant", "smoothquant_alpha": 0.8})
        assert config.activation_calibrator == "max"
        assert config.to_dict() == {"method": "smoothquant", "smoothquant_alpha": 0.8}

    def test_knob_must_match_method(self):
        with pytest.raises(ValueError, match="do not apply"):
            CalibrationConfig.from_raw({"method": "mse", "percentile": 99.0})
        with pytest.raises(ValueError, match="do not apply"):
            CalibrationConfig.from_raw({"method": "max", "smoothquant_alpha": 0.5})

    def test_unknown_method_and_bounds(self):
        with pytest.raises(ValueError, match="method must be one of"):
            CalibrationConfig.from_raw("kl")
        with pytest.raises(ValueError, match="percentile must be in"):
            CalibrationConfig.from_raw({"method": "percentile", "percentile": 0.0})
        with pytest.raises(ValueError, match="smoothquant_alpha must be in"):
            CalibrationConfig.from_raw({"method": "smoothquant", "smoothquant_alpha": 1.5})
        with pytest.raises(ValueError, match="quantization.calibration"):
            CalibrationConfig.from_raw({"method": "mse", "methd": "mse"})

    def test_travels_through_the_quantization_config(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "calibration": {"method": "percentile", "percentile": 99.99}}
        )
        assert config.calibration.method == "percentile"
        assert QuantizationConfig.from_dict(config.to_dict()) == config


class TestPTQConfig:
    def test_calibrate_samples_required(self):
        with pytest.raises(ValueError, match="calibrate_samples"):
            PTQConfig.from_dict({"batch_size": 1})

    def test_typo_guard(self):
        with pytest.raises(ValueError, match="Unknown quantization.ptq"):
            PTQConfig.from_dict({"calibrate_samples": 400, "batchsize": 2})


class TestQATConfig:
    def test_epochs_and_lr_required(self):
        with pytest.raises(ValueError, match="epochs"):
            QATConfig.from_dict({"lr": 1e-4})
        with pytest.raises(ValueError, match="lr"):
            QATConfig.from_dict({"epochs": 3})

    def test_typo_guard(self):
        with pytest.raises(ValueError, match="Unknown quantization.qat"):
            QATConfig.from_dict({"epochs": 3, "lr": 1e-4, "work_dir": "/tmp"})

    def test_schedule_defaults_to_cosine(self):
        schedule = QATConfig.from_dict({"epochs": 3, "lr": 1e-5}).schedule
        assert schedule.type == "cosine" and schedule.final_lr_ratio == 0.01

    def test_schedule_accepts_type_string(self):
        assert (
            QATConfig.from_dict({"epochs": 3, "lr": 1e-5, "schedule": "constant"}).schedule.type
            == "constant"
        )
        assert (
            QATConfig.from_dict({"epochs": 3, "lr": 1e-5, "schedule": "one_cycle"}).schedule.type
            == "one_cycle"
        )

    def test_schedule_accepts_mapping_with_type_knobs(self):
        schedule = QATConfig.from_dict(
            {
                "epochs": 3,
                "lr": 1e-4,
                "schedule": {"type": "one_cycle", "pct_start": 0.3, "cycle_momentum": True},
            }
        ).schedule
        assert (
            schedule.type == "one_cycle"
            and schedule.pct_start == 0.3
            and schedule.cycle_momentum is True
        )

    def test_freeze_unquantized_defaults_true(self):
        assert QATConfig.from_dict({"epochs": 3, "lr": 1e-4}).freeze_unquantized is True
        assert (
            QATConfig.from_dict(
                {"epochs": 3, "lr": 1e-4, "freeze_unquantized": False}
            ).freeze_unquantized
            is False
        )

    def test_val_check_interval_default_and_bounds(self):
        assert QATConfig.from_dict({"epochs": 3, "lr": 1e-4}).val_check_interval == 0.25
        with pytest.raises(ValueError, match="val_check_interval"):
            QATConfig.from_dict({"epochs": 3, "lr": 1e-4, "val_check_interval": 0.0})

    def test_schedule_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="schedule.type"):
            QATConfig.from_dict({"epochs": 3, "lr": 1e-4, "schedule": "linear"})


class TestQATScheduleConfig:
    def test_knob_must_match_type(self):
        with pytest.raises(ValueError, match="do not apply to type 'cosine'"):
            QATScheduleConfig.from_raw({"type": "cosine", "pct_start": 0.3})
        with pytest.raises(ValueError, match="Unknown quantization.qat.schedule key"):
            QATScheduleConfig.from_raw({"type": "cosine", "warmup": 1})

    def test_value_bounds(self):
        with pytest.raises(ValueError, match="final_lr_ratio"):
            QATScheduleConfig.from_raw({"type": "cosine", "final_lr_ratio": 0.0})
        with pytest.raises(ValueError, match="pct_start"):
            QATScheduleConfig.from_raw({"type": "one_cycle", "pct_start": 1.5})
        with pytest.raises(ValueError, match="div_factor"):
            QATScheduleConfig.from_raw({"type": "one_cycle", "div_factor": 0.5})

    def test_cosine_builds_no_warmup_one_cycle(self):
        scheduler, scheduler_config = QATScheduleConfig.from_raw(
            "cosine"
        ).build_lightning_scheduler(1e-5)
        assert scheduler["_target_"] == "torch.optim.lr_scheduler.OneCycleLR"
        assert scheduler["max_lr"] == 1e-5
        assert scheduler["div_factor"] == 1.0 and scheduler["pct_start"] == 0.0
        assert scheduler["final_div_factor"] == 100.0 and scheduler["cycle_momentum"] is False
        assert "total_steps" not in scheduler  # auto-filled by the optimizer builder
        assert scheduler_config == {"interval": "step"}

    def test_one_cycle_passes_reference_knobs(self):
        scheduler, _ = QATScheduleConfig.from_raw(
            {"type": "one_cycle", "cycle_momentum": True}
        ).build_lightning_scheduler(1e-4)
        assert scheduler["div_factor"] == 10.0 and scheduler["pct_start"] == 0.4
        assert scheduler["final_div_factor"] == 1.0e4
        assert scheduler["cycle_momentum"] is True and scheduler["base_momentum"] == 0.85

    def test_constant_has_no_scheduler(self):
        assert QATScheduleConfig.from_raw("constant").build_lightning_scheduler(1e-5) == (
            None,
            None,
        )

    def test_one_cycle_matches_torch_curve(self):
        torch = pytest.importorskip("torch")
        scheduler_cfg, _ = QATScheduleConfig.from_raw("cosine").build_lightning_scheduler(1e-5)
        kwargs = {k: v for k, v in scheduler_cfg.items() if not k.startswith("_")}
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, **kwargs)
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert lrs[0] == pytest.approx(1e-5, rel=1e-3)  # starts at the peak: no warmup
        assert lrs[50] == pytest.approx(4.9e-6, rel=0.05)  # halfway down the cosine
        assert lrs[99] == pytest.approx(1e-7, rel=0.05)  # anneals to 1% of the peak
