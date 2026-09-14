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

"""Shell-completion callbacks for the ``autoware-ml`` commands.

One thin, documented wrapper per completable parameter kind; the actual candidate
discovery lives in :mod:`autoware_ml.utils.cli.helpers`.
"""

from __future__ import annotations

import click

from autoware_ml.utils.cli.helpers import (
    EXPERIMENT_CONFIG_PREFIX,
    TASK_CONFIG_PREFIX,
    complete_config_value,
    complete_path_value,
    complete_session_command_value,
    complete_session_name_value,
)


def complete_task_config(incomplete: str) -> list[str]:
    """Complete task config names and config file paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for bundled task configs and YAML config paths.
    """
    return complete_config_value(incomplete, TASK_CONFIG_PREFIX)


def complete_experiment_config(incomplete: str) -> list[str]:
    """Complete experiment config names and config file paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for bundled experiment configs and YAML config paths.
    """
    return complete_config_value(incomplete, EXPERIMENT_CONFIG_PREFIX)


def complete_checkpoint_path(incomplete: str) -> list[str]:
    """Complete checkpoint file paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates limited to checkpoint files.
    """
    return complete_path_value(incomplete, file_suffixes=(".ckpt",))


def complete_directory_path(incomplete: str) -> list[str]:
    """Complete directory paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates limited to directories.
    """
    return complete_path_value(incomplete, directories_only=True)


def complete_any_path(incomplete: str) -> list[str]:
    """Complete generic filesystem paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for files and directories.
    """
    return complete_path_value(incomplete)


def complete_session_command(ctx: click.Context, incomplete: str) -> list[str]:
    """Complete commands forwarded through ``session start``.

    Args:
        ctx: Typer shell-completion context with parsed parameters.
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for the forwarded command line.
    """
    command_args = list(ctx.params.get("command_args", ()))
    return complete_session_command_value(command_args, incomplete)


def complete_session_name(incomplete: str) -> list[str]:
    """Complete managed session names.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for managed session names.
    """
    return complete_session_name_value(incomplete)
