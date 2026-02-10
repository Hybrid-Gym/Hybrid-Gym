"""Run inference for swe_doc_gen_locate benchmark using generic Python image.

This version uses python:3.11-bookworm instead of SWE-Gym Docker images,
cloning the repository at runtime. No specialized environment needed.

Task: Agent must LOCATE the function (no file path given) using the description,
then add a docstring.
"""
import asyncio
import copy
import json
import os
import tempfile
from typing import Any, Literal

import pandas as pd
import numpy as np
import toml
from datasets import load_dataset

import openhands.agenthub
from evaluation.benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
MASK_FUNC_NAME = os.environ.get('MASK_FUNC_NAME', 'false').lower() == 'true'
# ADDITIONAL_DETAILS: comma-separated list of detail types to include
# Valid values: 'parameters', 'returns', 'call_details' (or empty for none)
ADDITIONAL_DETAILS = [x.strip() for x in os.environ.get('ADDITIONAL_DETAILS', '').split(',') if x.strip()]
BenchMode = Literal['swe', 'swt', 'swt-ci']


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def updated_remove_binary_files_from_git():
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


def to_json_safe_dict(series: pd.Series):
    safe_dict = {}
    for k, v in series.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        try:
            json.dumps(v)
            safe_dict[k] = v
        except (TypeError, ValueError):
            continue
    return safe_dict


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    # Use base_commit as version identifier for SWE-Gym-Raw instances
    version = getattr(instance, 'version', None) or getattr(instance, 'base_commit', 'latest')
    if version is None:
        version = 'latest'
    return f'{instance.repo}__{version[:8]}'.replace('/', '__')


def _format_targets(functions: list[dict], mask_name: bool = False) -> str:
    """Format target functions with descriptions for Task 1 (locate).

    Args:
        functions: List of function dictionaries with module_name, module_type, function_description
        mask_name: If True, hide the function name to test agent's understanding of functionality
    """
    lines = []
    for idx, func in enumerate(functions, start=1):
        if mask_name:
            entry = f"{idx}. [MASKED] ({func.get('module_type', 'function')})"
        else:
            entry = f"{idx}. {func.get('module_name', '<unknown>')} ({func.get('module_type', 'function')})"
        description = func.get('function_description')
        if description:
            entry += f"\n   Description: {description}"
        lines.append(entry)
    return '\n'.join(lines)


def _format_additional_details(instance: pd.Series, detail_types: list[str]) -> str:
    """Format additional details (parameters, returns, call_details) for the prompt.

    Args:
        instance: The data instance containing detailed information
        detail_types: List of detail types to include ('parameters', 'returns', 'call_details')

    Returns:
        Formatted string with additional details, or empty string if none requested
    """
    if not detail_types:
        return ""

    sections = []

    if 'parameters' in detail_types:
        params = getattr(instance, 'parameters', None)
        if params:
            param_lines = []
            for p in params:
                param_str = f"  - {p.get('name', 'unknown')}"
                if p.get('type'):
                    param_str += f" ({p['type']})"
                if p.get('description'):
                    param_str += f": {p['description']}"
                if p.get('default') is not None:
                    param_str += f" [default: {p['default']}]"
                param_lines.append(param_str)
            if param_lines:
                sections.append("<parameters>\n" + "\n".join(param_lines) + "\n</parameters>")

    if 'returns' in detail_types:
        returns = getattr(instance, 'returns', None)
        if returns:
            return_str = ""
            if returns.get('type'):
                return_str += f"Type: {returns['type']}"
            if returns.get('description'):
                if return_str:
                    return_str += "\n"
                return_str += f"Description: {returns['description']}"
            if return_str:
                sections.append("<returns>\n" + return_str + "\n</returns>")

    if 'call_details' in detail_types:
        calls = getattr(instance, 'call_details', None)
        if calls:
            call_lines = []
            for c in calls:
                call_str = f"  - {c.get('function', 'unknown')}"
                if c.get('purpose'):
                    call_str += f": {c['purpose']}"
                call_lines.append(call_str)
            if call_lines:
                sections.append("<call_details>\n" + "\n".join(call_lines) + "\n</call_details>")

    if sections:
        return "\n\n<additional_details>\n" + "\n\n".join(sections) + "\n</additional_details>"
    return ""


def get_docstring_init_command(instance: pd.Series) -> str:
    """Generate shell command to remove existing docstrings."""
    # Support multi-function instances (functions list) and single-function instances
    if hasattr(instance, 'functions') and instance.functions:
        commands = []
        for func in instance.functions:
            file_path = func['file_path']
            docstring_start = func['docstring_line_start']
            docstring_end = func['docstring_line_end']
            if docstring_start != -1:
                commands.append(f"sed -i '{docstring_start+1},{docstring_end+1}d' {file_path}")
        return ' && '.join(commands) if commands else ""

    # Single function instance
    file_path = instance.file_path
    docstring_start, docstring_end = instance.docstring_line_start, instance.docstring_line_end
    return f"sed -i '{docstring_start+1},{docstring_end+1}d' {file_path}" if docstring_start != -1 else ""


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    """Generate instruction for locate task - agent must SEARCH for the function.

    When MASK_FUNC_NAME=true, the function name is hidden from the prompt to test
    whether the agent can locate functions based purely on functionality description.
    """
    # Use stored workspace dir name if available (set during initialize_runtime)
    # This ensures consistency since base_commit may be modified after git re-init
    workspace_dir_name = instance.get("_workspace_dir_name") or _get_swebench_workspace_dir_name(instance)

    if hasattr(instance, 'functions') and instance.functions:
        # Multi-function instance
        targets = instance.functions
        target_list = _format_targets(targets, mask_name=MASK_FUNC_NAME)
        instruction_body = f"""I've uploaded a python code repository in the directory {workspace_dir_name}.
Your goal is to locate the following {len(targets)} targets (docstrings removed) and write docstrings for each:

{target_list}

Follow these steps:
1. EXPLORATION: Search the codebase (file paths not provided) to find where each target is defined.
2. UNDERSTANDING: Read implementations to fully understand arguments, return values, and side effects.
3. RECHECK: Verify that the function/class you located actually matches the given description. Compare the implementation behavior with the description to ensure you found the correct target.
4. GENERATION: Insert a clear, concise Python docstring inside each definition using triple quotes. Cover purpose, parameters, return value(s), and important behaviors or exceptions.
5. REVIEW: Double-check formatting and accuracy; avoid changing code outside the docstrings unless strictly necessary.
Be thorough in your exploration and reasoning. Quality over brevity."""
    else:
        # Single function instance
        description = (
            getattr(instance, 'brief_description', None) or  # For masked data
            getattr(instance, 'function_description', None) or
            getattr(instance, 'description', None) or ''
        )
        if not description:
            description = 'No description provided.'

        if MASK_FUNC_NAME:
            # Masked mode: don't reveal the function name
            # Format additional details if ADDITIONAL_DETAILS env var is set
            additional_details_str = _format_additional_details(instance, ADDITIONAL_DETAILS)

            instruction_body = f"""I've uploaded a python code repository in the directory {workspace_dir_name}.
Your goal is to locate the {instance.module_type} described below and write its docstring.

<{instance.module_type}_description>
{description}
</{instance.module_type}_description>{additional_details_str}

Known details:
- Target type: {instance.module_type}
- The original docstring was removed; please restore/author it.
- Note: The target name is not provided. You must identify the correct {instance.module_type} based on the functionality description above.

Follow these steps:
1. EXPLORATION: Search the codebase (no file path provided) to find {instance.module_type}s that match the described functionality.
2. UNDERSTANDING: Read the implementation to fully understand its arguments, return values, and side effects.
3. RECHECK: Verify that the {instance.module_type} you located actually matches the given description. Compare the implementation behavior with the description to ensure you found the correct target.
4. GENERATION: Insert a clear, concise Python docstring inside the definition using triple quotes. Cover purpose, parameters, return value(s), and important behaviors or exceptions.
5. REVIEW: Double-check formatting and accuracy; avoid changing code outside the docstring unless strictly necessary.
Be thorough in your exploration and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity."""
        else:
            # Normal mode: reveal the function name
            instruction_body = f"""I've uploaded a python code repository in the directory {workspace_dir_name}.
Your goal is to locate the {instance.module_type} described below and write its docstring.

<{instance.module_type}_description>
{description}
</{instance.module_type}_description>

Known details:
- Target name: {instance.module_name}
- The original docstring was removed; please restore/author it.

Follow these steps:
1. EXPLORATION: Search the codebase (no file path provided) to find where this {instance.module_type} is defined.
2. UNDERSTANDING: Read the implementation to fully understand its arguments, return values, and side effects.
3. RECHECK: Verify that the {instance.module_type} you located actually matches the given description. Compare the implementation behavior with the description to ensure you found the correct target.
4. GENERATION: Insert a clear, concise Python docstring inside the definition using triple quotes. Cover purpose, parameters, return value(s), and important behaviors or exceptions.
5. REVIEW: Double-check formatting and accuracy; avoid changing code outside the docstring unless strictly necessary.
Be thorough in your exploration and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity."""

    instruction = f"""
<uploaded_files>
/workspace/{workspace_dir_name}
</uploaded_files>
{instruction_body}
"""

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    if hasattr(instance, 'image_assets') and instance.image_assets:
        assets = json.loads(instance.image_assets)
        assert 'problem_statement' in assets, (
            'problem_statement is required in image_assets'
        )
        image_urls = assets['problem_statement']
        return MessageAction(content=instruction, image_urls=image_urls)
    return MessageAction(content=instruction)


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'ghcr.io/all-hands-ai/runtime:0.40-nikolaik'
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
):
    """Initialize the runtime: clone repo, checkout commit, remove docstrings."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    # Store the workspace dir name so complete_runtime can use it
    # (base_commit gets modified later in this function)
    instance["_workspace_dir_name"] = workspace_dir_name
    obs: CmdOutputObservation

    # Set instance id and git configuration
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export SWE_INSTANCE_ID and configure git: {str(obs)}',
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    action = CmdRunAction(command='source ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if isinstance(obs, ErrorObservation):
        logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
    assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

    # git clone the repository
    action = CmdRunAction(command=f'git clone https://github.com/{instance["repo"]}.git /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git clone {instance["repo"]}: {str(obs)}')

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    # checkout to the base_commit
    action = CmdRunAction(command=f'git checkout {instance["base_commit"]}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git checkout {instance["base_commit"]}: {str(obs)}')

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    # Check python is available
    action = CmdRunAction(command='which python')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Expected to find python interpreter, but got: {str(obs)}',
    )

    # Remove docstring(s)
    cmd = get_docstring_init_command(instance)
    if cmd:
        action = CmdRunAction(command=cmd)
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to remove docstring: {str(obs)}')

    # Re-init git repository to track changes
    action = CmdRunAction(command='rm -rf .git')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to rm -rf .git: {str(obs)}')

    action = CmdRunAction(command='git init .')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git init: {str(obs)}')

    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert_and_raise(obs.exit_code == 0, f'Failed to git add -A: {str(obs)}')

    action = CmdRunAction(command='git commit -m "Initial commit"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert_and_raise(obs.exit_code == 0, f'Failed to git commit: {str(obs)}')

    # Get the new HEAD commit
    cmd = 'git log -1 --pretty="%H"'
    action = CmdRunAction(command=cmd)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to get git head commit: {str(obs)}')
    head_commit = obs.content.strip()
    instance["base_commit"] = head_commit

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and get git patch."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    # Use stored workspace dir name (base_commit was modified during initialization)
    workspace_dir_name = instance.get("_workspace_dir_name") or _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        logger.info('The previous command is still running, trying to ctrl+z it...')
        action = CmdRunAction(command='C-z')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    # Remove nested .git directories
    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to find git repositories: {str(obs)}',
    )

    git_dirs = [p for p in obs.content.strip().split('\n') if p]
    if git_dirs:
        for git_dir in git_dirs:
            action = CmdRunAction(command=f'rm -rf "{git_dir}"')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(
                isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                f'Failed to remove git directory {git_dir}: {str(obs)}',
            )

    # Add all files
    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    # Remove binary files
    action = CmdRunAction(command=updated_remove_binary_files_from_git())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                action = FileReadAction(path='patch.diff')
                action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                if isinstance(obs, FileReadObservation):
                    git_patch = obs.content
                    break
                elif isinstance(obs, ErrorObservation):
                    assert 'File could not be decoded as utf-8' in obs.content
                    action = CmdRunAction(command='cat patch.diff')
                    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                    git_patch = obs.content
                    break
                else:
                    assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')
    git_patch = remove_binary_diffs(git_patch)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance, metadata)
        message_action = get_instruction(instance, metadata)

        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()

    test_result = {
        'git_patch': git_patch,
    }

    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    instruction = message_action.content
    if message_action.image_urls:
        instruction += (
            '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
        )
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=to_json_safe_dict(instance),
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if 'selected_ids' in data:
                selected_ids = data['selected_ids']
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset
            if 'selected_repos' in data:
                selected_repos = data['selected_repos']
                if isinstance(selected_repos, str):
                    selected_repos = [selected_repos]
                assert isinstance(selected_repos, list)
                logger.info(
                    f'Filtering {selected_repos} tasks from "selected_repos"...'
                )
                subset = dataset[dataset['repo'].isin(selected_repos)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset

    skip_ids = os.environ.get('SKIP_IDS', '').split(',')
    if len(skip_ids) > 0:
        logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
        return dataset[~dataset[filter_column].isin(skip_ids)]
    return dataset


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='hybrid-gym/hybrid_gym_func_localize',
        help='data set to evaluate on (HuggingFace dataset name or local JSONL file path)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation",
    )

    args, _ = parser.parse_known_args()

    # Support loading from local JSONL file or HuggingFace dataset
    if args.dataset.endswith('.jsonl') or args.dataset.endswith('.json'):
        # Load from local file
        import json
        data = []
        with open(args.dataset, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        swe_bench_tests = pd.DataFrame(data)
        logger.info(f'Loaded {len(swe_bench_tests)} instances from local file: {args.dataset}')
    else:
        # Load from HuggingFace
        dataset = load_dataset(args.dataset, split=args.split)
        swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        try:
            llm_config.log_completions = True
            llm_config.modify_params = False
        except AttributeError:
            pass

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    details = {'mode': args.mode}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        condenser_config=condenser_config,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')

    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=8 * 60 * 60,
        max_retries=5,
    )
