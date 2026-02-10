"""Run inference for hybrid_gym_func_gen benchmark using generic Python image.

This version uses python:3.11-bookworm instead of specialized Docker images,
cloning the repository at runtime. No specialized environment needed.

Task: Agent is given a function signature and docstring (body removed),
and must implement the function body. Evaluation uses RepoST's eval_script.
"""
import ast
import asyncio
import copy
import json
import os
import re
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

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
# Whether to include original context in the prompt
INCLUDE_CONTEXT = os.environ.get('INCLUDE_CONTEXT', 'false').lower() == 'true'
# Whether to include return type/value information in the prompt
INCLUDE_RETURNS = os.environ.get('INCLUDE_RETURNS', 'false').lower() == 'true'
# Whether to include generated implementation instructions in the prompt (default: true)
INCLUDE_INSTRUCTION = os.environ.get('INCLUDE_INSTRUCTION', 'true').lower() == 'true'

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


def _get_workspace_dir_name(instance: pd.Series) -> str:
    """Generate workspace directory name from repo and commit."""
    version = getattr(instance, 'base_commit', 'latest')
    if version is None:
        version = 'latest'
    return f'{instance.repo}__{version[:8]}'.replace('/', '__')


def get_func_mask_command(instance: pd.Series) -> str:
    """Generate Python script to mask the function body, keeping signature and docstring.

    This creates a Python script that:
    1. Reads the source file
    2. Finds the target function (handles both regular functions and class methods)
    3. Removes the body but keeps signature and docstring
    4. Replaces body with 'pass  # TODO: Implement this function'
    """
    file_path = instance.file_path
    func_name = instance.func_name

    # Create a Python script to mask the function
    # This is more reliable than sed for Python code
    mask_script = f'''
import ast
import sys

def find_function_node(tree, func_name):
    """Find function node, handling both regular functions and class methods.

    Args:
        tree: AST tree
        func_name: Function name, can be 'func_name' or 'ClassName.method_name'

    Returns:
        The function/method AST node, or None if not found
    """
    # Check if it's a class method (contains a dot)
    if '.' in func_name:
        class_name, method_name = func_name.split('.', 1)
        # Find the class first
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Find the method inside the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return item
        return None
    else:
        # Regular function - search at module level first, then everywhere
        # First try module-level functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    return node
        # Fall back to searching everywhere (nested functions)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    return node
        return None

def find_function_and_mask(source_code, func_name):
    """Find function and return masked version."""
    lines = source_code.split('\\n')

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        print(f"Warning: Could not parse file, returning original", file=sys.stderr)
        return source_code

    # Find the function (handles both regular functions and class methods)
    func_node = find_function_node(tree, func_name)

    if func_node is None:
        print(f"Warning: Could not find function {{func_name}}", file=sys.stderr)
        return source_code

    # Get function start and end lines (1-indexed in AST)
    func_start = func_node.lineno - 1  # Convert to 0-indexed
    func_end = func_node.end_lineno  # end_lineno is already the last line (1-indexed)

    # Find where the body starts (after signature and docstring)
    body_start = func_start + 1  # Start after def line

    # Handle multi-line signature
    sig_line = lines[func_start]
    sig_line_idx = func_start
    while not sig_line.rstrip().endswith(':'):
        sig_line_idx += 1
        if sig_line_idx < len(lines):
            sig_line = lines[sig_line_idx]
        else:
            break
    body_start = sig_line_idx + 1

    # Check for docstring
    if func_node.body and isinstance(func_node.body[0], ast.Expr):
        if isinstance(func_node.body[0].value, ast.Constant) and isinstance(func_node.body[0].value.value, str):
            # Has docstring - body starts after docstring
            docstring_end = func_node.body[0].end_lineno
            body_start = docstring_end

    # Get indentation of function body
    if body_start < len(lines):
        first_body_line = lines[body_start]
        indent = len(first_body_line) - len(first_body_line.lstrip())
    else:
        # Default indentation
        def_line = lines[func_start]
        def_indent = len(def_line) - len(def_line.lstrip())
        indent = def_indent + 4

    indent_str = ' ' * indent

    # Create masked version
    new_lines = lines[:body_start]
    new_lines.append(f'{{indent_str}}pass  # TODO: Implement this function')
    new_lines.extend(lines[func_end:])

    return '\\n'.join(new_lines)

# Read the file
with open("{file_path}", "r") as f:
    source = f.read()

# Mask the function
masked = find_function_and_mask(source, "{func_name}")

# Write back
with open("{file_path}", "w") as f:
    f.write(masked)

print(f"Masked function {func_name} in {file_path}")
'''

    # Escape the script for shell
    escaped_script = mask_script.replace("'", "'\"'\"'")

    return f"python3 -c '{escaped_script}'"


def extract_return_info(instance: pd.Series) -> str:
    """Extract return type and example from function signature and body.

    Returns a formatted string with return information, or empty string if none found.
    """
    import re

    return_info_parts = []

    # 1. Extract return type from signature (e.g., "-> Path" or "-> int")
    signature = getattr(instance, 'func_signature', '') or ''
    return_type_match = re.search(r'->\s*([^:]+):', signature)
    if return_type_match:
        return_type = return_type_match.group(1).strip()
        return_info_parts.append(f"Return type: {return_type}")

    # 2. Extract return statements from body
    body = getattr(instance, 'func_body', '') or ''
    if body:
        return_statements = []
        for line in body.split('\n'):
            stripped = line.strip()
            if stripped.startswith('return '):
                # Get the return expression
                return_expr = stripped[7:].strip()
                if return_expr and return_expr not in return_statements:
                    return_statements.append(return_expr)

        if return_statements:
            # Show up to 2 unique return examples
            examples = return_statements[:2]
            if len(examples) == 1:
                return_info_parts.append(f"Returns: {examples[0]}")
            else:
                return_info_parts.append(f"Return examples: {', '.join(examples)}")

    if return_info_parts:
        return '\n'.join(return_info_parts)
    return ''


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    """Generate instruction for function completion task.

    The agent is given:
    - File path where the function is located
    - Function name
    - Function signature (visible in the file)
    - Docstring describing what the function should do (visible in the file)
    - The body is replaced with 'pass  # TODO: Implement this function'
    """
    workspace_dir_name = instance.get("_workspace_dir_name") or _get_workspace_dir_name(instance)

    file_path = instance.file_path
    func_name = instance.func_name

    # Get docstring for the prompt
    docstring_raw = getattr(instance, 'func_docstring_raw', '') or ''
    if not docstring_raw:
        docstring_raw = '(No docstring available - implement based on function name and context)'

    # Optionally extract return info
    return_info = ''
    if INCLUDE_RETURNS:
        return_info = extract_return_info(instance)

    # Build function_info section
    function_info_lines = [
        f"- File: {file_path}",
        f"- Function: {func_name}",
        f"- Docstring: {docstring_raw}",
    ]
    if return_info:
        function_info_lines.append(f"- {return_info.replace(chr(10), chr(10) + '  ')}")  # Indent multi-line

    function_info = '\n'.join(function_info_lines)

    # Build additional info sections (to place together after function_info)
    additional_sections = ""

    # Optionally include file imports (extracted from context)
    if INCLUDE_INSTRUCTION:
        orig_context = getattr(instance, 'orig_context', '')
        if orig_context:
            import_lines = []
            for line in orig_context.split('\n')[:50]:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    import_lines.append(stripped)
            if import_lines:
                imports_text = '\n'.join(import_lines)
                additional_sections += f"""
<file_imports>
The file uses these imports (use them as reference for your implementation):
{imports_text}
</file_imports>
"""

    # Optionally include implementation guidance
    if INCLUDE_INSTRUCTION:
        impl_instruction = getattr(instance, 'implementation_instruction', '')
        if impl_instruction:
            additional_sections += f"""
<implementation_guidance>
Here are some hints on how to implement this function:
{impl_instruction}
</implementation_guidance>
"""

    # Build instruction
    instruction_body = f"""I've uploaded a python code repository in the directory {workspace_dir_name}.

Your task is to implement the body of the function `{func_name}` in the file `{file_path}`.

The function signature and docstring are already in the file. The function body has been replaced with `pass  # TODO: Implement this function`.

<function_info>
{function_info}
</function_info>
{additional_sections}
Follow these steps:
1. NAVIGATE: Open the file `{file_path}` and locate the function `{func_name}`
2. CONTEXT: Read the surrounding code to understand:
   - What imports are available
   - What other functions/classes exist in the file
   - The coding style and patterns used
3. IMPLEMENT: Replace the `pass  # TODO: Implement this function` with a complete implementation that:
   - Fulfills the description in the docstring
   - Follows the function signature (parameters and return type if specified)
   - Uses appropriate imports and dependencies from the file
   - Handles edge cases appropriately
4. VERIFY: Review your implementation for:
   - Correctness (does it do what the docstring says?)
   - Completeness (all code paths handled?)
   - Style consistency (matches the rest of the codebase?)

Write clean, working Python code. Your implementation will be tested for correctness."""

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
    """Initialize the runtime: clone repo, checkout commit, mask function body."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_workspace_dir_name(instance)
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

    # Mask the function body (keep signature and docstring)
    mask_cmd = get_func_mask_command(instance)
    action = CmdRunAction(command=mask_cmd)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to mask function body: {str(obs)}')

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

    action = CmdRunAction(command='git commit -m "Initial commit with masked function"')
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


def extract_function_from_file(file_content: str, func_name: str) -> str:
    """Extract a specific function's source code from file content.

    Handles both regular functions and class methods (e.g., 'ClassName.method_name').

    Args:
        file_content: The full file content
        func_name: Name of the function to extract (can be 'func' or 'Class.method')

    Returns:
        The function source code, or empty string if not found
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return ""

    lines = file_content.split('\n')

    # Check if it's a class method (contains a dot)
    if '.' in func_name:
        class_name, method_name = func_name.split('.', 1)
        # Find the class first
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Find the method inside the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            func_lines = lines[item.lineno - 1:item.end_lineno]
                            return '\n'.join(func_lines)
        return ""
    else:
        # Regular function - search at module level first, then everywhere
        # First try module-level functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    func_lines = lines[node.lineno - 1:node.end_lineno]
                    return '\n'.join(func_lines)
        # Fall back to searching everywhere (nested functions)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    func_lines = lines[node.lineno - 1:node.end_lineno]
                    return '\n'.join(func_lines)
        return ""


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and extract the generated function."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = instance.get("_workspace_dir_name") or _get_workspace_dir_name(instance)

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

    # Get git patch
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

    # Read the completed function from the file
    file_path = instance.file_path
    action = FileReadAction(path=file_path)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    generated_function = ""
    if isinstance(obs, FileReadObservation):
        file_content = obs.content
        generated_function = extract_function_from_file(file_content, instance.func_name)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {
        'git_patch': git_patch,
        'generated_function': generated_function,
    }


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
        generated_function = return_val['generated_function']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
        logger.info(
            f'Generated function:\n--------\n{generated_function}\n--------'
        )
    finally:
        runtime.close()

    test_result = {
        'git_patch': git_patch,
        'generated_function': generated_function,
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
        default='hybrid-gym/hybrid_gym_func_gen',
        help='data set to evaluate on (HuggingFace dataset name or local JSONL file path)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )

    args, _ = parser.parse_known_args()

    # Support loading from local JSONL file
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

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    # Override eval_output_dir if EXPERIMENT_NAME is set
    # This writes directly to experiments/<name>/ instead of default evaluation_outputs/
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    if experiment_name:
        eval_output_dir = os.path.join(
            'evaluation/benchmarks/hybrid_gym_func_gen/experiments',
            experiment_name
        )
        logger.info(f'Using EXPERIMENT_NAME: {experiment_name}')
        logger.info(f'Output directory: {eval_output_dir}')
    else:
        eval_output_dir = args.eval_output_dir

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        eval_output_dir,
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
