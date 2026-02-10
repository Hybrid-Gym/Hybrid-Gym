"""Run inference for swe_bench_dep benchmark.

Task: Given a function, find all modules (functions/classes) directly called by it
and add a comment above each called module's definition.

Uses generic python:3.11-bookworm Docker image.

Usage:
    python run_infer.py \
        --llm-config your_llm_config \
        --agent-cls CodeActAgent \
        --dataset resource/swe_bench_dep_data.jsonl \
        --max-iterations 30 \
        --eval-num-workers 4
"""

import asyncio
import copy
import json
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

import openhands.agenthub
from evaluation.benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
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
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def updated_remove_binary_files_from_git():
    """Shell command to remove binary files from git staging."""
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


def to_json_safe_dict(series: pd.Series):
    """Convert pandas Series to JSON-safe dict."""
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


def _get_workspace_dir_name(instance: pd.Series) -> str:
    """Generate workspace directory name from instance."""
    commit = instance.base_commit[:8] if hasattr(instance, 'base_commit') else 'latest'
    return f'{instance.repo}__{commit}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    """Generate instruction for the dependency annotation task."""
    workspace_dir_name = _get_workspace_dir_name(instance)

    func_name = instance.target_function_name
    func_file = instance.target_function_file
    func_line = instance.target_function_line_start + 1  # 1-indexed for display

    instruction = f"""
<uploaded_files>
/workspace/{workspace_dir_name}
</uploaded_files>

I've uploaded a Python code repository in the directory `/workspace/{workspace_dir_name}`.

## Your Task

Analyze the function `{func_name}` located at **line {func_line}** in file `{func_file}` and find ALL modules (functions or classes) that are **directly called** by this function within this repository.

For each called module that is defined within this repository (not Python built-ins or external libraries), you need to add a comment **right above its definition**:

```python
# this function/class is called by the {func_name} function
def some_called_function():
    ...
```

## What Counts as a "Call"

The following patterns all count as calling a module:
- **Function calls**: `helper_function()`
- **Class instantiation**: `MyClass()` - this counts as calling `MyClass`
- **Exception raising**: `raise MyException()` - this counts as calling `MyException`

## Important Rules

1. **Only annotate modules that are DIRECTLY called** by `{func_name}` (not transitive dependencies)
2. **Only annotate modules defined WITHIN this repository** - ignore:
   - Python built-ins (print, len, range, str, list, dict, etc.)
   - Standard library functions
   - External package functions
3. **Place the comment on the line immediately ABOVE** the `def` or `class` keyword
4. **If there are decorators**, place the comment ABOVE the decorators
5. **Do NOT modify any code** other than adding these comments
6. The comment must say exactly: `# this function/class is called by the {func_name} function`
7. **Do NOT add duplicate comments** - each dependency should be annotated exactly once

## Recommended Steps

1. **READ**: First, read the function `{func_name}` in `{func_file}` to understand what it does
2. **ANALYZE**: Identify all function/class calls within the function body (typically 1-3 dependencies)
3. **RESOLVE EACH CALL**: For each call, determine which definition is actually in scope:
   - Check if it's defined locally in the same file (could be above OR below the target function)
   - Check the imports at the top of `{func_file}` to see if it's imported from another module
   - Follow the import to find the actual definition
4. **FILTER**: Exclude Python built-ins and external library calls
5. **ANNOTATE**: Add the comment above each qualifying definition in the repository
6. **VERIFY**: Double-check that all comments are correctly placed and there are no duplicates

Be thorough in your analysis. Quality and completeness are more important than speed.
"""

    if RUN_WITH_BROWSING:
        instruction += (
            '\n<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web.</IMPORTANT!>\n'
        )

    return MessageAction(content=instruction)


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Get OpenHands configuration for the benchmark."""
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'python:3.11-bookworm'
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
    """Initialize the runtime: clone repo, checkout commit, setup git."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_workspace_dir_name(instance)
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

    # Git clone the repository
    action = CmdRunAction(
        command=f'git clone https://github.com/{instance["repo"]}.git /workspace/{workspace_dir_name}'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to git clone {instance["repo"]}: {str(obs)}'
    )

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    # Checkout to the base_commit
    action = CmdRunAction(command=f'git checkout {instance["base_commit"]}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to git checkout {instance["base_commit"]}: {str(obs)}',
    )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    # Remove git remotes to prevent accidental pushes
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

    # Re-init git repository to track agent's changes from a clean state
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

    # Get the new HEAD commit for diff comparison
    cmd = 'git log -1 --pretty="%H"'
    action = CmdRunAction(command=cmd)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to get git head commit: {str(obs)}')
    head_commit = obs.content.strip()

    # Store the new base commit for diff in complete_runtime
    instance["_base_commit_for_diff"] = head_commit

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and extract git patch."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # Handle stuck commands
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

    # Remove binary files from git staging
    action = CmdRunAction(command=updated_remove_binary_files_from_git())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    # Get the base commit for diff
    base_commit = instance.get("_base_commit_for_diff", "HEAD~1")

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {base_commit} > patch.diff'
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

    # Remove binary diffs from the patch
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
    """Process a single instance."""
    config = get_config(instance, metadata)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource_factor with increasing attempt_id
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


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='hybrid-gym/hybrid_gym_dep_search',
        help='HuggingFace dataset name or local JSONL file path',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='split to evaluate on',
    )

    args, _ = parser.parse_known_args()

    # Support loading from local JSONL file or HuggingFace
    if args.dataset.endswith('.jsonl') or args.dataset.endswith('.json'):
        data = []
        with open(args.dataset, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        swe_bench_tests = pd.DataFrame(data)
        logger.info(f'Loaded {len(swe_bench_tests)} instances from local file: {args.dataset}')
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        swe_bench_tests = dataset.to_pandas()
        logger.info(f'Loaded {len(swe_bench_tests)} instances from HuggingFace: {args.dataset}')

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

    details = {}

    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_description,
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
        timeout_seconds=8 * 60 * 60,  # 8 hours per instance
        max_retries=5,
    )
