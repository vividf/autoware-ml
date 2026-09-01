"""Tests for the per-iteration timing callback: step metrics and their train-only
restriction, epoch summaries, warmup exclusion, and sanity-check suppression."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autoware_ml.callbacks.iteration_timer import IterationTimer


class _Clock:
    """Deterministic replacement for ``time.perf_counter``."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = iter(ticks)

    def __call__(self) -> float:
        return next(self._ticks)


def _module() -> MagicMock:
    module = MagicMock()
    module.log = MagicMock()
    return module


def _trainer(sanity_checking: bool = False) -> MagicMock:
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    return trainer


def _logged(module: MagicMock) -> dict[str, float]:
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


def _run_train_iters(
    callback: IterationTimer,
    trainer: MagicMock,
    module: MagicMock,
    ticks: list[float],
) -> None:
    """Drive whole train iterations from a flat list of start/end timestamps."""
    callback._now = _Clock(ticks)  # type: ignore[method-assign]
    for batch_idx in range(len(ticks) // 2):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=batch_idx)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=batch_idx)


class TestStepMetrics:
    def test_iteration_time_is_logged_per_step(self) -> None:
        callback = IterationTimer()
        module = _module()

        _run_train_iters(callback, _trainer(), module, [0.0, 1.5])

        assert _logged(module)["train/batch_forward_time"] == pytest.approx(1.5)

    def test_data_time_measures_gap_between_iterations(self) -> None:
        callback = IterationTimer()
        module = _module()

        # iter 0 runs 0.0 -> 1.0, iter 1 starts at 1.25 (0.25s fetching).
        _run_train_iters(callback, _trainer(), module, [0.0, 1.0, 1.25, 2.0])

        assert _logged(module)["train/data_waiting_time"] == pytest.approx(0.25)

    def test_first_iteration_has_no_data_time(self) -> None:
        callback = IterationTimer()
        module = _module()

        _run_train_iters(callback, _trainer(), module, [0.0, 1.0])

        assert "train/data_waiting_time" not in _logged(module)

    def test_stages_are_timed_independently(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        trainer = _trainer()
        callback._now = _Clock([0.0, 10.0, 100.0, 100.5])  # type: ignore[method-assign]

        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_validation_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_validation_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
        callback.on_validation_epoch_end(trainer, module)

        logged = _logged(module)
        # Validation is timed but only summarised; training keeps its step metric.
        assert logged["val/batch_forward_time_mean"] == pytest.approx(90.0)
        assert logged["train/batch_forward_time"] == pytest.approx(100.5)


class TestEvaluationStagesAreSummaryOnly:
    """Lightning flushes eval step metrics every batch, so they are not logged."""

    @pytest.mark.parametrize(
        ("start_hook", "end_hook"),
        [
            ("on_validation_batch_start", "on_validation_batch_end"),
            ("on_test_batch_start", "on_test_batch_end"),
        ],
    )
    def test_no_step_metrics_are_logged(self, start_hook: str, end_hook: str) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        trainer = _trainer()
        callback._now = _Clock([0.0, 1.0, 3.0, 5.0])  # type: ignore[method-assign]

        for batch_idx in (0, 1):
            getattr(callback, start_hook)(trainer, module, batch=None, batch_idx=batch_idx)
            getattr(callback, end_hook)(
                trainer, module, outputs=None, batch=None, batch_idx=batch_idx
            )

        assert module.log.call_args_list == []

    def test_the_epoch_summary_still_covers_every_iteration(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        trainer = _trainer()
        callback._now = _Clock([0.0, 1.0, 3.0, 5.0])  # type: ignore[method-assign]

        for batch_idx in (0, 1):
            callback.on_validation_batch_start(trainer, module, batch=None, batch_idx=batch_idx)
            callback.on_validation_batch_end(
                trainer, module, outputs=None, batch=None, batch_idx=batch_idx
            )
        callback.on_validation_epoch_end(trainer, module)

        logged = _logged(module)
        assert logged["val/batch_forward_time_mean"] == pytest.approx(1.5)
        assert logged["val/batch_forward_time_max"] == pytest.approx(2.0)
        assert logged["val/data_waiting_time_mean"] == pytest.approx(2.0)
        assert logged["val/total_iter_time_mean"] == pytest.approx(4.0)


class TestTotalIterationTime:
    def test_total_is_the_fetch_plus_the_forward(self) -> None:
        callback = IterationTimer()
        module = _module()

        # iter 0 runs 0.0 -> 1.0, iter 1 fetches for 0.25s then runs for 0.75s.
        _run_train_iters(callback, _trainer(), module, [0.0, 1.0, 1.25, 2.0])

        logged = _logged(module)
        assert logged["train/data_waiting_time"] == pytest.approx(0.25)
        assert logged["train/batch_forward_time"] == pytest.approx(0.75)
        assert logged["train/total_iter_time"] == pytest.approx(1.0)

    def test_first_iteration_has_no_total(self) -> None:
        callback = IterationTimer()
        module = _module()

        _run_train_iters(callback, _trainer(), module, [0.0, 1.0])

        assert "train/total_iter_time" not in _logged(module)

    def test_epoch_summary_reports_mean_and_max(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        trainer = _trainer()

        # Totals of the 2nd and 3rd iterations: (1.0 + 1.0) and (2.0 + 3.0).
        _run_train_iters(callback, trainer, module, [0.0, 10.0, 11.0, 12.0, 14.0, 17.0])
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        assert logged["train/total_iter_time_mean"] == pytest.approx(3.5)
        assert logged["train/total_iter_time_max"] == pytest.approx(5.0)


class TestEpochSummary:
    def test_summary_excludes_warmup_iterations(self) -> None:
        callback = IterationTimer(warmup_iters=1)
        module = _module()
        trainer = _trainer()

        # Iterations of 10s, 1s, 3s: the 10s warmup must not reach the summary.
        _run_train_iters(callback, trainer, module, [0.0, 10.0, 10.0, 11.0, 11.0, 14.0])
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        assert logged["train/batch_forward_time_mean"] == pytest.approx(2.0)
        assert logged["train/batch_forward_time_max"] == pytest.approx(3.0)

    def test_summary_is_skipped_when_all_iterations_are_warmup(self) -> None:
        callback = IterationTimer(warmup_iters=5)
        module = _module()
        trainer = _trainer()

        _run_train_iters(callback, trainer, module, [0.0, 1.0])
        callback.on_train_epoch_end(trainer, module)

        assert "train/batch_forward_time_mean" not in _logged(module)

    def test_epoch_end_resets_state_for_next_epoch(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        trainer = _trainer()

        _run_train_iters(callback, trainer, module, [0.0, 10.0])
        callback.on_train_epoch_end(trainer, module)
        module.log.reset_mock()

        _run_train_iters(callback, trainer, module, [100.0, 101.0])
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        # A stale batch-end timestamp would show up as a 99s data_time.
        assert "train/data_waiting_time" not in logged
        assert logged["train/batch_forward_time_mean"] == pytest.approx(1.0)


class TestSanityCheck:
    def test_sanity_check_iterations_are_not_recorded(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        sanity = _trainer(sanity_checking=True)
        callback._now = _Clock([])  # type: ignore[method-assign]

        callback.on_validation_batch_start(sanity, module, batch=None, batch_idx=0)
        callback.on_validation_batch_end(sanity, module, outputs=None, batch=None, batch_idx=0)
        callback.on_validation_epoch_end(sanity, module)

        assert module.log.call_args_list == []

    def test_timing_resumes_cleanly_after_sanity_check(self) -> None:
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        sanity = _trainer(sanity_checking=True)
        callback._now = _Clock([])  # type: ignore[method-assign]
        callback.on_validation_batch_start(sanity, module, batch=None, batch_idx=0)
        callback.on_validation_epoch_end(sanity, module)

        trainer = _trainer()
        callback._now = _Clock([0.0, 2.0])  # type: ignore[method-assign]
        callback.on_validation_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_validation_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
        callback.on_validation_epoch_end(trainer, module)

        assert _logged(module)["val/batch_forward_time_mean"] == pytest.approx(2.0)


class TestConstructorValidation:
    @pytest.mark.parametrize("kwargs", [{"warmup_iters": -1}, {"log_interval": -2}])
    def test_negative_values_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            IterationTimer(**kwargs)
