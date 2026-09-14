# Copyright 2025 TIER IV, Inc.
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

"""Main CLI entry point for Autoware-ML commands.

This module defines the Typer application, command groups, and shell
completion helpers used by the ``autoware-ml`` executable.
"""

import logging
from importlib.metadata import version
from pathlib import Path

import click
import typer
from click.core import ParameterSource
from click.shell_completion import CompletionItem
from typer.core import TyperCommand
from typing_extensions import Annotated

from autoware_ml.cli.completion import (
    complete_any_path,
    complete_checkpoint_path,
    complete_directory_path,
    complete_experiment_config,
    complete_session_command,
    complete_session_name,
    complete_task_config,
)
from autoware_ml.utils.cli.helpers import (
    EXPERIMENT_CONFIG_PREFIX,
    TASK_CONFIG_PREFIX,
    parse_extra_args,
    resolve_config_reference,
    run_lazy_script,
)

app = typer.Typer(
    name="autoware-ml",
    help="Autoware-ML - Machine learning framework for Autoware",
    no_args_is_help=True,
    add_completion=True,
)
mlflow_app = typer.Typer(
    name="mlflow",
    help="MLflow utilities",
    no_args_is_help=True,
)
session_app = typer.Typer(
    name="session",
    help="Managed background task sessions",
    no_args_is_help=True,
)

CONFIG_PREFIXES = (TASK_CONFIG_PREFIX, EXPERIMENT_CONFIG_PREFIX)
# TODO(vividf): drop this comment block together with the tasks/ family (design doc Q5).
# One command name, one implementation per config family, dispatched by config prefix:
# ``tasks/`` configs run the legacy single-task (BaseModel) entrypoints, kept because
# other models on main still train through them; ``experiments/`` configs run the
# multi-task (MultiTaskBaseModel + builders) entrypoints. Deployment and quantization
# exist only for the experiments family, which is why those two commands hard-require
# an experiments/ config.
#
# ``quantize`` + ``deploy`` replaced the old monolithic ``multi_task_deploy``:
# quantize produces a NEW self-describing checkpoint (PTQ / QAT — config + placement
# record embedded), deploy turns any checkpoint into inference artifacts (ONNX/TRT
# export + verification + evaluation). Different outputs, different lifecycles; the
# self-describing checkpoint is what lets the two stages stay decoupled.
TRAIN_ENTRYPOINT_MODULE = "autoware_ml.scripts.train"
EXPERIMENT_TRAIN_ENTRYPOINT_MODULE = "autoware_ml.scripts.experiment_train"
TEST_ENTRYPOINT_MODULE = "autoware_ml.scripts.test"
EXPERIMENT_TEST_ENTRYPOINT_MODULE = "autoware_ml.scripts.experiment_test"
DEPLOY_ENTRYPOINT_MODULE = "autoware_ml.scripts.deploy"
QUANTIZE_ENTRYPOINT_MODULE = "autoware_ml.scripts.quantize"
ENTRYPOINT_MODULES = {
    ("train", TASK_CONFIG_PREFIX): TRAIN_ENTRYPOINT_MODULE,
    ("train", EXPERIMENT_CONFIG_PREFIX): EXPERIMENT_TRAIN_ENTRYPOINT_MODULE,
    ("test", TASK_CONFIG_PREFIX): TEST_ENTRYPOINT_MODULE,
    ("test", EXPERIMENT_CONFIG_PREFIX): EXPERIMENT_TEST_ENTRYPOINT_MODULE,
    ("deploy", EXPERIMENT_CONFIG_PREFIX): DEPLOY_ENTRYPOINT_MODULE,
    ("quantize", EXPERIMENT_CONFIG_PREFIX): QUANTIZE_ENTRYPOINT_MODULE,
}
CLI_RUNTIME_MODULE = "autoware_ml.cli.runtime"


class OptionFirstTyperCommand(TyperCommand):
    """Suggest command options even when completion starts on an empty token."""

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Return shell completions with options prioritized for empty tokens.

        Args:
            ctx: Active Click command context.
            incomplete: Current incomplete shell token.

        Returns:
            Completion candidates for the current command line.
        """
        results = super().shell_complete(ctx, incomplete)
        if incomplete:
            return results

        seen = {item.value for item in results}
        for param in self.get_params(ctx):
            if (
                not isinstance(param, click.Option)
                or param.hidden
                or (
                    not param.multiple
                    and ctx.get_parameter_source(param.name) is ParameterSource.COMMANDLINE
                )
            ):
                continue
            for option_name in [*param.opts, *param.secondary_opts]:
                if option_name in seen:
                    continue
                results.append(CompletionItem(option_name, help=param.help))
                seen.add(option_name)
        return results


def setup_logging(level: str = "INFO") -> None:
    """Configure process-wide logging for CLI execution.

    Args:
        level: Root logging level name.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


@app.callback(invoke_without_command=True)
def main_callback(
    version_flag: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
) -> None:
    """Handle top-level CLI options before subcommand execution.

    Args:
        version_flag: Whether to print the installed package version and exit.
    """
    setup_logging()
    if version_flag:
        typer.echo(f"autoware-ml {version('autoware-ml')}")
        raise typer.Exit()


def resolve_config_prefix(config_name: str, default_prefix: str) -> str:
    """Pick the config family (``tasks`` / ``experiments``) a config reference belongs to.

    An explicit ``tasks/...`` or ``experiments/...`` prefix decides; a bundled path or a
    YAML file under ``autoware_ml/configs/<family>/`` decides; anything else (a bare
    name, an external YAML) falls back to the command's default family.

    Args:
        config_name: Config name or YAML path as given on the command line.
        default_prefix: Family assumed when the reference carries none.

    Returns:
        The config prefix to hand to the runtime.
    """
    for prefix in CONFIG_PREFIXES:
        if config_name.startswith(f"{prefix}/"):
            return prefix
    _, resolved_name, _ = resolve_config_reference(config_name, default_prefix)
    for prefix in CONFIG_PREFIXES:
        if resolved_name.startswith(f"{prefix}/"):
            return prefix
    return default_prefix


def resolve_entrypoint(command: str, config_name: str, default_prefix: str) -> tuple[str, str]:
    """Resolve ``(entrypoint_module, config_prefix)`` for a command and config reference.

    Raises:
        typer.BadParameter: When the config family has no implementation of the command
            (deploy / quantize exist only for ``experiments/`` configs).
    """
    config_prefix = resolve_config_prefix(config_name, default_prefix)
    entrypoint = ENTRYPOINT_MODULES.get((command, config_prefix))
    if entrypoint is None:
        supported = sorted(prefix for cmd, prefix in ENTRYPOINT_MODULES if cmd == command)
        raise typer.BadParameter(
            f"'{command}' is not available for {config_prefix}/ configs "
            f"(supported: {', '.join(f'{p}/' for p in supported)}). Got --config-name {config_name!r}."
        )
    return entrypoint, config_prefix


def _weights_override(weights: list[str]) -> str:
    return "+weights=[" + ",".join(weights) + "]"


@app.command(
    name="train",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Config name or YAML config path (tasks/... or experiments/...)",
            autocompletion=complete_task_config,
        ),
    ],
    weights: Annotated[
        list[str] | None,
        typer.Option(
            "--weights",
            help="One or more checkpoint paths for pretrained weight initialization "
            "(repeatable; later checkpoints overwrite earlier ones). "
            "Mutually exclusive with --resume-checkpoint.",
            autocompletion=complete_checkpoint_path,
        ),
    ] = None,
    resume_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--resume-checkpoint",
            help="Full Lightning checkpoint path to resume training from "
            "(restores model weights, optimizer state, and epoch, and continues "
            "the checkpoint's source MLflow run). Mutually exclusive with --weights.",
            autocompletion=complete_checkpoint_path,
        ),
    ] = None,
    new_run: Annotated[
        bool,
        typer.Option(
            "--new-run",
            help="With --resume-checkpoint: continue the training state in a new "
            "MLflow run instead of the checkpoint's source run.",
        ),
    ] = False,
) -> None:
    """Run model training through the Hydra-backed training entrypoint.

    Pass ``--weights`` to initialize model parameters from one or more pretrained
    checkpoints before training starts (e.g. transfer learning from a seg3d backbone
    into a det3d model). Pass ``--resume-checkpoint`` to resume an interrupted training
    run from its full saved state; it continues inside the checkpoint's source MLflow
    run unless ``--new-run`` forks it. The two options are mutually exclusive.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Config name or config file path to train.
        weights: One or more checkpoint paths for pretrained weight initialization.
        resume_checkpoint: Full Lightning checkpoint path to resume training from.
        new_run: Whether to fork the resumed training into a new MLflow run.
    """
    if weights and resume_checkpoint:
        raise typer.BadParameter("--weights and --resume-checkpoint are mutually exclusive.")
    if new_run and not resume_checkpoint:
        raise typer.BadParameter("--new-run requires --resume-checkpoint.")

    hydra_overrides: list[str] = []
    if weights:
        hydra_overrides.append(_weights_override(weights))
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise typer.BadParameter(f"Resume checkpoint '{resume_checkpoint}' does not exist.")
        resume_checkpoint = str(resume_path)
        hydra_overrides.append(f"+resume_checkpoint={resume_checkpoint}")

    entrypoint_module, config_prefix = resolve_entrypoint("train", config_name, TASK_CONFIG_PREFIX)
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=entrypoint_module,
        config_name=config_name,
        stage="train",
        extra_args=ctx.args,
        hydra_overrides=hydra_overrides,
        resume_checkpoint=resume_checkpoint,
        new_run=new_run,
        config_prefix=config_prefix,
    )


@app.command(
    name="deploy",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def deploy(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Experiment config name or YAML config path (experiments/...)",
            autocompletion=complete_experiment_config,
        ),
    ],
    weights: Annotated[
        list[str] | None,
        typer.Option(
            "--weights",
            help="One or more checkpoint paths to merge into the deployed model "
            "(repeatable; later checkpoints overwrite earlier ones). A quantized "
            "(PTQ/QAT) checkpoint is detected from its embedded description.",
            autocompletion=complete_checkpoint_path,
        ),
    ] = None,
    release: Annotated[
        str | None,
        typer.Option(
            "--release",
            help="Release stamped into the ONNX metadata (vMAJOR.MINOR.PATCH, e.g. "
            "v0.0.1). Omitting it marks the artifacts 'unversioned' — fine for quick "
            "tests, never for production.",
        ),
    ] = None,
) -> None:
    """Export, verify, and evaluate a trained model (ONNX / TensorRT).

    The run exports one artifact per stage the model declares, checks cross-backend
    numerical parity (``deploy.verification``), and scores every enabled backend against
    ground truth under the same metric keys as ``test`` (``deploy.evaluation``).

    Every exported ONNX module is stamped with its identity and provenance
    (producer, release, config, commits, class lists). Pass ``--release`` for
    anything that may reach production; without it the artifacts are stamped
    ``unversioned`` and the deploy logs a warning.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Experiment config name or config file path to deploy.
        weights: One or more checkpoint paths to merge into the deployed model.
        release: Release stamped into the ONNX metadata; None marks the export unversioned.
    """
    if not weights:
        raise typer.BadParameter("--weights <path> (repeatable) must be specified.")

    hydra_overrides = [_weights_override(weights)]
    if release is not None:
        hydra_overrides.append(f"+release={release}")

    entrypoint_module, config_prefix = resolve_entrypoint(
        "deploy", config_name, EXPERIMENT_CONFIG_PREFIX
    )
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=entrypoint_module,
        config_name=config_name,
        stage="deploy",
        extra_args=ctx.args,
        hydra_overrides=hydra_overrides,
        checkpoints=weights,
        config_prefix=config_prefix,
    )


@app.command(
    name="quantize",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def quantize(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Experiment config name or YAML config path (experiments/...)",
            autocompletion=complete_experiment_config,
        ),
    ],
    weights: Annotated[
        list[str] | None,
        typer.Option(
            "--weights",
            help="One or more FP checkpoint paths to quantize "
            "(repeatable; later checkpoints overwrite earlier ones)",
            autocompletion=complete_checkpoint_path,
        ),
    ] = None,
) -> None:
    """Produce a self-describing quantized checkpoint (PTQ or QAT) from an FP checkpoint.

    The config's ``quantization`` section selects the mode: ``ptq`` calibrates on the
    validation split and saves ``ptq.ckpt``; ``qat`` runs frozen-amax fine-tuning and
    saves ``best.ckpt``/``last.ckpt``. The produced checkpoint embeds the quantization
    description, so ``deploy`` and ``test`` need no quantization config to load it.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Experiment config name or config file path to quantize with.
        weights: One or more FP checkpoint paths providing the model weights.
    """
    if not weights:
        raise typer.BadParameter("--weights <path> (repeatable) must be specified.")

    entrypoint_module, config_prefix = resolve_entrypoint(
        "quantize", config_name, EXPERIMENT_CONFIG_PREFIX
    )
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=entrypoint_module,
        config_name=config_name,
        stage="quantize",
        extra_args=ctx.args,
        hydra_overrides=[_weights_override(weights)],
        checkpoints=weights,
        config_prefix=config_prefix,
    )


@app.command(
    name="test",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def test(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Config name or YAML config path (tasks/... or experiments/...)",
            autocompletion=complete_task_config,
        ),
    ],
    weights: Annotated[
        list[str] | None,
        typer.Option(
            "--weights",
            help="One or more checkpoint paths to load into the model for evaluation "
            "(repeatable; later checkpoints overwrite earlier ones). A quantized "
            "(PTQ/QAT) checkpoint is detected from its embedded description.",
            autocompletion=complete_checkpoint_path,
        ),
    ] = None,
    use_config_devices: Annotated[
        bool,
        typer.Option(
            "--use-config-devices",
            help="Evaluate on the trainer.devices from the config. By default test forces a "
            "single device for deterministic evaluation that avoids distributed-sampler padding.",
        ),
    ] = False,
) -> None:
    """Run model evaluation through the Hydra-backed test entrypoint.

    Pass ``--weights`` once per checkpoint that should contribute parameters to
    the evaluated model. Every parameter must be covered by at least one checkpoint;
    multi-task evaluation stacks multiple ``--weights`` to merge independently
    trained heads into one model.

    By default evaluation runs on a single device, which is deterministic and free of
    the distributed-sampler padding that slightly skews multi-GPU metrics. Pass
    ``--use-config-devices`` to honor ``trainer.devices`` from the config instead.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Config name or config file path to evaluate.
        weights: One or more checkpoint paths to load into the model for evaluation.
        use_config_devices: Keep the config's ``trainer.devices`` instead of forcing one device.
    """
    if not weights:
        raise typer.BadParameter("--weights <path> (repeatable) must be specified.")

    hydra_overrides = [_weights_override(weights)]
    if not use_config_devices:
        # Applied after the user's extra args, so it wins: test defaults to one device.
        hydra_overrides.append("++trainer.devices=1")
    primary_checkpoint = weights[-1]

    entrypoint_module, config_prefix = resolve_entrypoint("test", config_name, TASK_CONFIG_PREFIX)
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=entrypoint_module,
        config_name=config_name,
        stage="test",
        extra_args=ctx.args,
        hydra_overrides=hydra_overrides,
        checkpoint=primary_checkpoint,
        config_prefix=config_prefix,
    )


@mlflow_app.command(name="ui", cls=OptionFirstTyperCommand)
def mlflow_ui(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to listen on")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 5000,
    db_path: Annotated[
        str,
        typer.Option(
            "--db-path",
            help="Path to SQLite backend store file",
            autocompletion=complete_any_path,
        ),
    ] = "mlruns/mlflow.db",
) -> None:
    """Launch MLflow UI against a selected backend store.

    Args:
        host: Host interface used by the MLflow UI server.
        port: TCP port used by the MLflow UI server.
        db_path: Path to the SQLite backend store file.
    """
    run_lazy_script(
        "autoware_ml.scripts.mlflow_wrapper",
        "run_mlflow_ui",
        host=host,
        port=port,
        db_path=db_path,
    )


@mlflow_app.command(name="export", cls=OptionFirstTyperCommand)
def mlflow_export(
    db_path: Annotated[
        str,
        typer.Option(
            "--db-path",
            help="Path to SQLite backend store file",
            autocompletion=complete_any_path,
        ),
    ] = "mlruns/mlflow.db",
    config_name: Annotated[
        str | None,
        typer.Option(
            "--config-name",
            help="Export the experiment matching this task config path",
            autocompletion=complete_task_config,
        ),
    ] = None,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", help="Export only this MLflow experiment")
    ] = None,
    export_dir: Annotated[
        str | None,
        typer.Option(
            "--export-dir",
            help="Directory for the extracted experiment store",
            autocompletion=complete_directory_path,
        ),
    ] = None,
    override: Annotated[
        bool,
        typer.Option("--override", help="Allow replacing an existing exported MLflow store"),
    ] = False,
) -> None:
    """Export one MLflow experiment into an isolated tracking store.

    Args:
        config_name: User-facing config name used to derive the experiment name.
        experiment_name: Explicit MLflow experiment name to export.
        export_dir: Output directory for the exported tracking store.
        db_path: Path to the source SQLite backend store file.
        override: Whether to overwrite an existing export directory.
    """
    run_lazy_script(
        "autoware_ml.scripts.mlflow_wrapper",
        "export_experiment_from_db",
        db_path=db_path,
        experiment_name=experiment_name,
        config_name=config_name,
        export_dir=export_dir,
        override=override,
    )


@app.command(
    name="create-dataset",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def create_dataset(
    ctx: typer.Context,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset name (e.g., nuscenes, nuscenes_mini)"),
    ],
    task: Annotated[list[str], typer.Option("--task", help="Task name (can be repeated)")],
    root_path: Annotated[
        str,
        typer.Option(
            "--root-path",
            help="Root path of the dataset",
            autocompletion=complete_directory_path,
        ),
    ],
    out_dir: Annotated[
        str,
        typer.Option(
            "--out-dir",
            help="Output directory for info files",
            autocompletion=complete_directory_path,
        ),
    ],
) -> None:
    """Generate dataset info files with specified tasks.

    Requires dataset name and at least one task.
    """

    run_lazy_script(
        "autoware_ml.scripts.create_dataset",
        "main",
        dataset=dataset,
        tasks=task,
        root_path=root_path,
        out_dir=out_dir,
        **parse_extra_args(ctx.args),
    )


@session_app.command(
    name="start",
    cls=OptionFirstTyperCommand,
)
def session_start(
    name: Annotated[str, typer.Option("--name", "-n", help="Session name")],
    cwd: Annotated[
        str | None,
        typer.Option(
            "--cwd",
            help="Working directory for the session command",
            autocompletion=complete_directory_path,
        ),
    ] = None,
    attach: Annotated[
        bool, typer.Option("--attach", help="Open the live viewer immediately after starting")
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Run the forwarded command as-is instead of prefixing it with autoware-ml",
        ),
    ] = False,
    command_args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Command to run in the managed background session. Pass it after '--', e.g. -- train --config-name ...",
            autocompletion=complete_session_command,
        ),
    ] = None,
) -> None:
    """Start a detached managed session for a background task.

    Args:
        name: Managed session name.
        cwd: Working directory used when launching the session command.
        attach: Whether to open the live viewer immediately after startup.
        raw: Whether to execute the forwarded command directly instead of
            prefixing it with ``autoware-ml``.
        command_args: Command tokens forwarded to the managed shell.
    """
    run_lazy_script(
        "autoware_ml.scripts.session",
        "start_session",
        name=name,
        command_args=command_args or [],
        cwd=cwd,
        attach=attach,
        raw=raw,
    )
    if not attach:
        typer.echo(f"Started session '{name}'.")
        typer.echo(f"View live output with: autoware-ml session attach --name {name}")
        typer.echo("Press Ctrl+C in the viewer to return without stopping the task.")
        typer.echo(f"Stop the task with: autoware-ml session stop --name {name}")


@session_app.command(name="attach", cls=OptionFirstTyperCommand)
def session_attach(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Render a live terminal view of a managed session.

    Args:
        name: Name of the session to view.
    """
    run_lazy_script("autoware_ml.scripts.session", "attach_session", name=name)


@session_app.command(name="detach", cls=OptionFirstTyperCommand)
def session_detach(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Disconnect raw tmux clients from a managed session.

    Args:
        name: Name of the session whose tmux clients should be detached.
    """
    run_lazy_script("autoware_ml.scripts.session", "detach_session", name=name)


@session_app.command(name="ls", cls=OptionFirstTyperCommand)
def session_ls() -> None:
    """List background sessions managed by ``autoware-ml``.

    The command prints formatted session information and exits quietly when no
    managed sessions are currently running.
    """
    output = run_lazy_script("autoware_ml.scripts.session", "list_sessions")
    if output:
        typer.echo(output)


@session_app.command(name="stop", cls=OptionFirstTyperCommand)
def session_stop(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Stop a managed background task and close its session.

    Args:
        name: Name of the session to stop.
    """
    run_lazy_script("autoware_ml.scripts.session", "stop_session", name=name)


def main() -> None:
    """Run the top-level Typer application.

    This wrapper keeps the installed entrypoint and ``python -m`` execution
    path aligned on the same CLI startup logic.
    """
    app()


app.add_typer(mlflow_app, name="mlflow")
app.add_typer(session_app, name="session")


if __name__ == "__main__":
    main()
