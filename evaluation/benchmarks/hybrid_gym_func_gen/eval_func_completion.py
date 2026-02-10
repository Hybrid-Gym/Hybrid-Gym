"""Evaluate function completion results using RepoST's eval_script.

This script takes the output.jsonl from inference and evaluates each
generated function by running the eval_script tests.

Usage:
    python eval_func_completion.py \
        --output-file outputs/my-experiment/.../output.jsonl \
        --use-docker  # or --no-docker for direct execution
        --single-container  # (default) run all tests in one container for speed
        --no-single-container  # run each test in a separate container
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


@contextmanager
def docker_container(docker_image: str = 'yiqingxyq/repost:v0'):
    """Context manager to start and stop a persistent Docker container.

    Usage:
        with docker_container('yiqingxyq/repost:v0') as container_id:
            run_test_in_running_container(test_content, container_id)
    """
    container_id = None
    try:
        # Generate a unique container name
        container_name = f'repost_eval_{uuid.uuid4().hex[:8]}'

        # Start container in detached mode with tail -f /dev/null to keep it running
        cmd = [
            'docker', 'run', '-d',
            '--platform=linux/arm64',
            '--name', container_name,
            docker_image,
            'tail', '-f', '/dev/null'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f'Failed to start Docker container: {result.stderr}')

        container_id = result.stdout.strip()
        print(f'Started Docker container: {container_name} ({container_id[:12]})')
        yield container_id

    finally:
        if container_id:
            # Stop and remove the container
            subprocess.run(
                ['docker', 'stop', container_id],
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ['docker', 'rm', container_id],
                capture_output=True,
                timeout=30
            )
            print(f'Stopped and removed Docker container: {container_id[:12]}')


def run_test_in_running_container(
    test_content: str,
    container_id: str,
    timeout: int = 60
) -> dict:
    """Run test inside a running Docker container using docker exec.

    This is faster than docker run because the container is already running.

    Args:
        test_content: The test file content
        container_id: ID of the running container
        timeout: Timeout in seconds

    Returns:
        Dictionary with 'success', 'output', 'error' keys
    """
    # Create a unique test file name to avoid conflicts
    test_filename = f'/tmp/test_{uuid.uuid4().hex[:8]}.py'

    try:
        # Write test content to container using docker exec with heredoc via bash
        # Escape special characters for shell
        escaped_content = test_content.replace('\\', '\\\\').replace("'", "'\"'\"'")

        # Write the file inside the container
        write_cmd = [
            'docker', 'exec', container_id,
            'bash', '-c', f"cat > {test_filename} << 'EOFTEST'\n{test_content}\nEOFTEST"
        ]
        write_result = subprocess.run(write_cmd, capture_output=True, text=True, timeout=30)
        if write_result.returncode != 0:
            return {
                'success': False,
                'output': '',
                'error': f'Failed to write test file: {write_result.stderr}',
                'returncode': -1
            }

        # Run the test
        run_cmd = [
            'docker', 'exec', container_id,
            'python', test_filename
        ]
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )

        success = result.returncode == 0
        return {
            'success': success,
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': 'Timeout expired',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'returncode': -1
        }
    finally:
        # Clean up test file inside container
        cleanup_cmd = ['docker', 'exec', container_id, 'rm', '-f', test_filename]
        subprocess.run(cleanup_cmd, capture_output=True, timeout=10)


def extract_function_body(generated_function: str, func_name: str) -> str:
    """Extract just the function body (without signature/docstring) from generated function.

    Args:
        generated_function: The full generated function source
        func_name: Name of the function

    Returns:
        The function body only
    """
    import ast

    try:
        tree = ast.parse(generated_function)
        if not tree.body:
            return generated_function

        func_node = tree.body[0]
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return generated_function

        # Get the docstring end line if exists
        body_start_line = 1  # After the def line
        if func_node.body:
            first_stmt = func_node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                if isinstance(first_stmt.value.value, str):
                    # Has docstring
                    body_start_line = first_stmt.end_lineno

        lines = generated_function.split('\n')
        body_lines = lines[body_start_line:]
        return '\n'.join(body_lines)
    except SyntaxError:
        return generated_function


def rename_function_in_code(code: str, old_name: str, new_name: str) -> str:
    """Rename a function definition in code.

    Args:
        code: The source code
        old_name: Original function name
        new_name: New function name

    Returns:
        Code with function renamed
    """
    import re
    # Replace 'def old_name(' with 'def new_name('
    # Also handle 'async def old_name('
    pattern = rf'((?:async\s+)?def\s+){old_name}(\s*\()'
    return re.sub(pattern, rf'\1{new_name}\2', code)


def construct_test_file(
    generated_function: str,
    eval_script: str,
    func_name: str
) -> str:
    """Construct a test file combining the generated function with eval_script.

    The eval_script structure:
    1. Mock implementations / imports
    2. The ORIGINAL function implementation (named func_name)
    3. Test functions that compare func_name(input) vs func_name_new_implementation(input)

    For standalone functions:
    - Rename the agent's generated function to {func_name}_new_implementation
    - Insert it AFTER the imports but BEFORE the original function

    For class methods (func_name contains '.', e.g., 'ClassName.method_name'):
    - Rename just the method_name to method_name_new_implementation
    - Insert it INSIDE the class, right before the original method
    - Preserve any decorators (@staticmethod, @classmethod, etc.)

    Args:
        generated_function: The agent's generated function
        eval_script: The RepoST eval_script containing tests
        func_name: Name of the function

    Returns:
        Complete test file content
    """
    # Check if it's a class method (func_name contains '.')
    if '.' in func_name:
        return _construct_test_file_for_method(generated_function, eval_script, func_name)
    else:
        return _construct_test_file_for_function(generated_function, eval_script, func_name)


def _construct_test_file_for_function(
    generated_function: str,
    eval_script: str,
    func_name: str
) -> str:
    """Construct test file for standalone functions (original logic)."""
    # Rename the generated function to {func_name}_new_implementation
    new_impl_name = f'{func_name}_new_implementation'
    renamed_function = rename_function_in_code(generated_function, func_name, new_impl_name)

    # Extract imports from eval_script and place them first
    # This ensures type annotations in generated function can reference imported types
    lines = eval_script.split('\n')
    import_lines = []
    other_lines = []
    in_imports = True

    for line in lines:
        stripped = line.strip()
        # Check if line is an import, comment, or empty (part of header)
        if in_imports and (stripped.startswith('import ') or
                          stripped.startswith('from ') or
                          stripped.startswith('#') or
                          stripped == ''):
            import_lines.append(line)
        else:
            in_imports = False
            other_lines.append(line)

    imports_section = '\n'.join(import_lines)
    rest_of_script = '\n'.join(other_lines)

    # Build test file: imports -> generated function -> rest of eval_script
    test_file = f'''{imports_section}

# Agent's generated function (renamed to {new_impl_name})
{renamed_function}

# Original eval_script (original implementation and tests)
{rest_of_script}
'''
    return test_file


def _construct_test_file_for_method(
    generated_function: str,
    eval_script: str,
    func_name: str
) -> str:
    """Construct test file for class methods.

    For class methods like 'BFSCluster.forward':
    1. Rename 'forward' -> 'forward_new_implementation' in generated code
    2. Find and preserve decorators (@staticmethod, @classmethod)
    3. Insert the new method inside the class, right before the original method
    4. Ensure correct indentation (4 spaces for class methods)
    """
    class_name, method_name = func_name.rsplit('.', 1)
    new_method_name = f"{method_name}_new_implementation"

    # Step 1: Rename the method in generated code
    renamed_generated = re.sub(
        rf'((?:async\s+)?def\s+){method_name}(\s*\()',
        rf'\1{new_method_name}\2',
        generated_function
    )

    # Step 2: Ensure correct indentation (4 spaces for class methods)
    # Check current indentation of the generated function
    first_line = renamed_generated.split('\n')[0]
    current_indent = len(first_line) - len(first_line.lstrip())

    if current_indent != 4:
        # Adjust indentation to 4 spaces
        lines = renamed_generated.split('\n')
        adjusted_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                # Remove current indentation and add 4 spaces
                stripped = line.lstrip()
                # Calculate relative indentation from first line
                line_indent = len(line) - len(stripped)
                relative_indent = line_indent - current_indent
                new_indent = 4 + relative_indent
                adjusted_lines.append(' ' * new_indent + stripped)
            else:
                adjusted_lines.append(line)
        renamed_generated = '\n'.join(adjusted_lines)

    # Step 3: Find decorators on the original method in eval_script
    # Pattern matches decorators (lines starting with 4 spaces + @) followed by def
    decorator_pattern = rf'((?:    @\w+.*\n)*)    def {method_name}\s*\('
    decorator_match = re.search(decorator_pattern, eval_script)

    decorators = ""
    if decorator_match and decorator_match.group(1).strip():
        decorators = decorator_match.group(1)  # Keep the decorators with their indentation

    # Step 4: Add decorators to the new implementation if needed
    if decorators:
        renamed_generated = decorators + renamed_generated

    # Step 5: Find insertion point (right before original method definition, including its decorators)
    if decorator_match:
        insert_pos = decorator_match.start()
    else:
        # Fallback: find just the def (with 4-space indent for class method)
        method_match = re.search(rf'    def {method_name}\s*\(', eval_script)
        if method_match:
            insert_pos = method_match.start()
        else:
            # Last resort: can't find the method, fall back to function logic
            return _construct_test_file_for_function(generated_function, eval_script, func_name)

    # Step 6: Build new eval_script with inserted method inside the class
    new_eval_script = (
        eval_script[:insert_pos] +
        renamed_generated + "\n\n" +
        eval_script[insert_pos:]
    )

    # Step 7: Extract imports and build final test file
    lines = new_eval_script.split('\n')
    import_lines = []
    other_lines = []
    in_imports = True

    for line in lines:
        stripped = line.strip()
        # Check if line is an import, comment, or empty (part of header)
        if in_imports and (stripped.startswith('import ') or
                          stripped.startswith('from ') or
                          stripped.startswith('#') or
                          stripped == ''):
            import_lines.append(line)
        else:
            in_imports = False
            other_lines.append(line)

    imports_section = '\n'.join(import_lines)
    rest_of_script = '\n'.join(other_lines)

    test_file = f'''{imports_section}

# Modified eval_script with agent's {new_method_name} inserted into {class_name} class
{rest_of_script}
'''
    return test_file


def run_test_directly(test_content: str, timeout: int = 60) -> dict:
    """Run test directly using Python subprocess.

    Args:
        test_content: The test file content
        timeout: Timeout in seconds

    Returns:
        Dictionary with 'success', 'output', 'error' keys
    """
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(test_content)
        test_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        success = result.returncode == 0
        return {
            'success': success,
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': 'Timeout expired',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'returncode': -1
        }
    finally:
        os.unlink(test_file)


def run_test_in_docker(
    test_content: str,
    docker_image: str = 'yiqingxyq/repost:v0',
    timeout: int = 120
) -> dict:
    """Run test inside RepoST Docker container.

    Args:
        test_content: The test file content
        docker_image: Docker image to use
        timeout: Timeout in seconds

    Returns:
        Dictionary with 'success', 'output', 'error' keys
    """
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, dir='/tmp'
    ) as f:
        f.write(test_content)
        test_file = f.name

    try:
        # Run in Docker
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{test_file}:/home/user/test.py:ro',
            docker_image,
            'python', '/home/user/test.py'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        success = result.returncode == 0
        return {
            'success': success,
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': 'Timeout expired',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'returncode': -1
        }
    finally:
        os.unlink(test_file)


def evaluate_instance(
    entry: dict,
    use_docker: bool = True,
    docker_image: str = 'yiqingxyq/repost:v0',
    timeout: int = 60,
    container_id: Optional[str] = None
) -> dict:
    """Evaluate a single instance.

    Args:
        entry: A single entry from output.jsonl
        use_docker: Whether to run tests in Docker
        docker_image: Docker image to use (only used if container_id is None)
        timeout: Timeout for test execution
        container_id: Optional ID of a running container to use (for single-container mode)

    Returns:
        Evaluation result dictionary
    """
    instance_id = entry.get('instance_id', 'unknown')
    test_result = entry.get('test_result', {})
    instance = entry.get('instance', {})

    generated_function = test_result.get('generated_function', '')
    eval_script = instance.get('eval_script', '')
    func_name = instance.get('func_name', '')

    if not generated_function:
        return {
            'instance_id': instance_id,
            'success': False,
            'reason': 'no_generated_function',
            'output': '',
            'error': 'No generated function found'
        }

    if not eval_script:
        return {
            'instance_id': instance_id,
            'success': False,
            'reason': 'no_eval_script',
            'output': '',
            'error': 'No eval_script found in instance'
        }

    if not func_name:
        return {
            'instance_id': instance_id,
            'success': False,
            'reason': 'no_func_name',
            'output': '',
            'error': 'No func_name found in instance'
        }

    # Construct test file
    test_content = construct_test_file(generated_function, eval_script, func_name)

    # Run test
    if container_id:
        # Use running container (single-container mode)
        result = run_test_in_running_container(test_content, container_id, timeout)
    elif use_docker:
        # Create new container for each test (legacy mode)
        result = run_test_in_docker(test_content, docker_image, timeout)
    else:
        # Run directly without Docker
        result = run_test_directly(test_content, timeout)

    return {
        'instance_id': instance_id,
        'success': result['success'],
        'reason': 'passed' if result['success'] else 'test_failed',
        'output': result['output'],
        'error': result['error'],
        'returncode': result['returncode']
    }


def update_output_with_eval(
    output_file: str,
    use_docker: bool = True,
    docker_image: str = 'yiqingxyq/repost:v0',
    timeout: int = 60,
    backup: bool = True,
    single_container: bool = True
) -> str:
    """
    Update the output.jsonl file with evaluation results.

    Adds to each entry:
    - history_length: Number of steps in the agent's history
    - eval_result: Dict with success, reason, error

    Args:
        output_file: Path to output.jsonl from inference
        use_docker: Whether to run tests in Docker
        docker_image: Docker image to use for evaluation
        timeout: Timeout for each test in seconds
        backup: If True, create a backup of the original file
        single_container: If True, run all tests in a single Docker container (faster)

    Returns:
        Path to the updated output file
    """
    from tqdm import tqdm
    import shutil

    # Read all entries
    entries = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Create backup if requested
    if backup:
        backup_file = output_file + '.bak'
        if os.path.exists(backup_file):
            print(f"Backup file already exists: {backup_file}")
        else:
            shutil.copy(output_file, backup_file)
            print(f"Created backup: {backup_file}")

    # Update each entry with eval results
    updated_entries = []
    passed = 0
    failed = 0

    def process_entries(container_id=None):
        nonlocal passed, failed
        for entry in tqdm(entries, desc='Evaluating and updating output'):
            # Calculate history length
            history = entry.get('history', [])
            history_length = len(history)

            # Evaluate instance
            result = evaluate_instance(
                entry,
                use_docker=use_docker,
                docker_image=docker_image,
                timeout=timeout,
                container_id=container_id
            )

            if result['success']:
                passed += 1
            else:
                failed += 1

            # Add new fields to entry
            entry['history_length'] = history_length
            entry['eval_result'] = {
                'success': result['success'],
                'reason': result['reason'],
                'error': result.get('error', ''),
            }

            updated_entries.append(entry)

    # Run evaluation with or without single container
    if use_docker and single_container:
        with docker_container(docker_image) as container_id:
            process_entries(container_id)
    else:
        process_entries()

    # Write back to file
    with open(output_file, 'w') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + '\n')

    total = len(updated_entries)
    print(f"\nUpdated {total} entries in {output_file}")
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {failed}/{total} ({100*failed/total:.1f}%)" if total > 0 else "Failed: 0")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate function completion results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to output.jsonl from inference'
    )
    parser.add_argument(
        '--use-docker',
        action='store_true',
        default=True,
        help='Run tests inside Docker container (default: True)'
    )
    parser.add_argument(
        '--no-docker',
        action='store_true',
        default=False,
        help='Run tests directly without Docker'
    )
    parser.add_argument(
        '--docker-image',
        type=str,
        default='yiqingxyq/repost:v0',
        help='Docker image to use for evaluation'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout for each test in seconds'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Output file for evaluation results (default: {output-file}.eval.json)'
    )
    parser.add_argument(
        '--update-output',
        action='store_true',
        help='Update output.jsonl with history_length and eval_result fields'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup when using --update-output'
    )
    parser.add_argument(
        '--single-container',
        action='store_true',
        default=True,
        help='Run all tests in a single Docker container (default: True, faster)'
    )
    parser.add_argument(
        '--no-single-container',
        action='store_true',
        default=False,
        help='Run each test in a separate Docker container (slower, but allows parallel execution)'
    )

    args = parser.parse_args()

    # Handle --no-docker flag (overrides default)
    use_docker = not args.no_docker

    # Handle --no-single-container flag (overrides default)
    single_container = not args.no_single_container

    # Load output file
    output_path = Path(args.output_file)
    if not output_path.exists():
        print(f"Error: Output file not found: {args.output_file}")
        sys.exit(1)

    entries = []
    with open(output_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.output_file}")
    mode_str = 'directly (no Docker)' if not use_docker else ('single container' if single_container else 'separate containers')
    print(f"Running evaluation in {mode_str}")

    # Evaluate each entry
    results = []
    passed = 0
    failed = 0

    def run_evaluations(container_id=None):
        nonlocal passed, failed
        for i, entry in enumerate(entries):
            print(f"Evaluating {i+1}/{len(entries)}: {entry.get('instance_id', 'unknown')}...", end=' ')

            result = evaluate_instance(
                entry,
                use_docker=use_docker,
                docker_image=args.docker_image,
                timeout=args.timeout,
                container_id=container_id
            )
            results.append(result)

            if result['success']:
                passed += 1
                print("PASSED")
            else:
                failed += 1
                print(f"FAILED ({result['reason']})")

    # Run evaluation with or without single container
    if use_docker and single_container:
        with docker_container(args.docker_image) as container_id:
            run_evaluations(container_id)
    else:
        run_evaluations()

    # Summary
    total = len(results)
    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total instances: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)" if total > 0 else "Failed: 0")

    # Save results
    results_file = args.results_file or str(output_path) + '.eval.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total if total > 0 else 0
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print failures
    failures = [r for r in results if not r['success']]
    if failures:
        print("\n" + "=" * 50)
        print("FAILURES:")
        print("=" * 50)
        for f in failures[:10]:  # Show first 10 failures
            print(f"- {f['instance_id']}: {f['reason']}")
            if f['error']:
                print(f"  Error: {f['error'][:200]}...")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures")

    # Update output.jsonl with eval results if requested
    if args.update_output:
        print("\n=== Updating output.jsonl with eval results ===")
        update_output_with_eval(
            args.output_file,
            use_docker=use_docker,
            docker_image=args.docker_image,
            timeout=args.timeout,
            backup=not args.no_backup,
            single_container=single_container
        )


if __name__ == '__main__':
    main()
