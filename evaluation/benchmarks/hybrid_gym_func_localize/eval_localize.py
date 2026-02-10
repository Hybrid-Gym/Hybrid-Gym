import os
import json
import argparse
from datasets import load_dataset
import re
from collections import defaultdict
from tqdm import tqdm

STORAGE_DIR = os.environ.get('STORAGE_DIR', '/home/yiqingxi')


def patch2file_paths(patch):
    # get all file paths from the patch
    file_paths = set()

    # Parse git diff format to extract file paths
    # Look for lines like "diff --git a/path/to/file b/path/to/file"
    diff_lines = patch.split('\n')

    for line in diff_lines:
        if line.startswith('diff --git'):
            # Extract file path from "diff --git a/path/to/file b/path/to/file"
            parts = line.split()
            if len(parts) >= 4:
                # Remove the 'a/' prefix to get the actual file path
                file_path = parts[2][2:]  # Remove 'a/' prefix
                file_paths.add(file_path)

    return file_paths


def parse_git_patch(patch_text):
    """
    Parse a git patch and return a list of dictionaries for each modified file.

    Args:
        patch_text (str): The git patch text

    Returns:
        list: List of dictionaries, each containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)
    """
    # Split the patch into lines
    lines = patch_text.split('\n')

    files = []
    current_file = None
    added_lines = []
    removed_lines = []

    # Pattern to match file path in diff header
    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    for line in lines:
        # Extract filename from diff header
        if line.startswith('diff --git'):
            # Save previous file if exists
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_lines': added_lines,
                    'removed_lines': removed_lines
                })

                current_file = None
                added_lines = []
                removed_lines = []

            # Start new file
            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)  # Use the 'b/' path (new file)

        # Extract added lines (lines starting with +)
        elif line.startswith('+') and not line.startswith('+++') and current_file is not None:
            content = line[1:] if line.startswith('+') else line
            content = content.strip()
            if content != '':
                added_lines.append(content)

        # Extract removed lines (lines starting with -)
        elif line.startswith('-') and not line.startswith('---') and current_file is not None:
            content = line[1:] if line.startswith('-') else line
            content = content.strip()
            if content != '':
                removed_lines.append(content)

    # Don't forget the last file
    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_lines': added_lines,
            'removed_lines': removed_lines
        })

    return files


def is_lines_comment_only(lines: list[str]) -> bool:
    """
    Check if a list of lines contains only comments (# comments or docstrings).

    Handles:
    - Single-line # comments
    - Single-line docstrings: \"\"\"text\"\"\" or '''text'''
    - Multi-line docstrings across multiple lines
    - Empty lines / whitespace

    Args:
        lines: List of line strings (already stripped of diff +/- prefixes)

    Returns:
        bool: True if all lines are comments or docstrings, False otherwise
    """
    if not lines:
        return True

    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Empty lines are OK
        if not stripped:
            continue

        # If we're inside a multi-line docstring
        if in_docstring:
            # Check if this line ends the docstring
            if docstring_char in stripped:
                # Handle cases like: some text"""
                if stripped.endswith(docstring_char):
                    in_docstring = False
                    docstring_char = None
                # Handle cases where """ appears but line continues (edge case)
                elif stripped.count(docstring_char) % 2 == 1:
                    in_docstring = False
                    docstring_char = None
            continue

        # Check for # comments
        if stripped.startswith('#'):
            continue

        # Check for docstrings
        # Single-line docstring: """text""" or '''text'''
        for quote in ['"""', "'''"]:
            if stripped.startswith(quote):
                if stripped.count(quote) >= 2:
                    # Single-line docstring (opens and closes on same line)
                    # Check if content after the docstring is just whitespace or nothing
                    # Remove the docstring and check what's left
                    after_first = stripped[3:]
                    if quote in after_first:
                        # Has closing quote - it's a complete docstring
                        continue
                    else:
                        # Multi-line docstring starts
                        in_docstring = True
                        docstring_char = quote
                        break
                else:
                    # Multi-line docstring starts
                    in_docstring = True
                    docstring_char = quote
                    break
        else:
            # Not a comment, not a docstring start
            # Could be a continuation line inside docstring that doesn't contain quotes
            # But since we track in_docstring state, if we reach here and not in docstring,
            # it's a non-comment line
            if not in_docstring:
                return False

    return True


def check_add_comments_only(patch_dict):
    """
    Check if a patch dictionary only adds comments/docstrings and doesn't remove any lines.

    Args:
        patch_dict (dict): Dictionary containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)

    Returns:
        bool: True if only comments/docstrings were added, False otherwise
    """
    # Check if there are any removed lines
    has_removals = len(patch_dict['removed_lines']) > 0
    if has_removals:
        return False

    # Check if all added lines are comments or docstrings
    return is_lines_comment_only(patch_dict['added_lines'])


def test_parse_git_patch():
    # Example usage with the provided patch
    example_patch = \
"""diff --git a/monai/data/grid_dataset.py b/monai/data/grid_dataset.py
index fc8175f6..ef426d72 100644
--- a/monai/data/grid_dataset.py
+++ b/monai/data/grid_dataset.py
@@ -67,6 +67,9 @@ class PatchIter:
             array: the image to generate patches from.

         \"\"\"
+        # TODO: This method uses iter_patch which has an issue with GPU tensors
+        # The iter_patch function converts coordinates to numpy array, which fails for GPU tensors
+        # This needs to be fixed in iter_patch to properly handle GPU tensors
         yield from iter_patch(
             array,
             patch_size=self.patch_size,  # type: ignore
@@ -189,6 +192,10 @@ class GridPatchDataset(IterableDataset):

     def __iter__(self):
         for image in super().__iter__():
+            # TODO: This method uses patch_iter which calls iter_patch
+            # The iter_patch function has an issue with GPU tensors because it converts coordinates to numpy
+            # This causes a failure when using GPU tensors as input
+            # The fix needs to be in iter_patch to properly handle GPU tensors
             for patch, *others in self.patch_iter(image):
                 out_patch = patch
                 if self.patch_transform is not None:
diff --git a/monai/data/utils.py b/monai/data/utils.py
index 5461fda9..7c3a8bd7 100644
--- a/monai/data/utils.py
+++ b/monai/data/utils.py
@@ -314,6 +314,11 @@ def iter_patch(
             coords_no_pad = tuple((coord.start - p, coord.stop - p) for coord, p in zip(slices, _pad_size))
         else:
             coords_no_pad = tuple((coord.start, coord.stop) for coord in slices)
+        # TODO: This line needs modification to support GPU tensors
+        # The current code uses np.asarray which doesn't work with GPU tensors
+        # We need to create the coordinates on the same device as the input array
+        # If arrpad is a PyTorch tensor, create coords as a tensor on the same device
+        # Otherwise, continue using numpy array
         yield arrpad[slices], np.asarray(coords_no_pad)  # data and coords (in numpy; works with torch loader)

     # copy back data from the padded image if required
"""

    # Test the function
    results = parse_git_patch(example_patch)
    print("Parsed patch result:")
    print(f"Number of modified files: {len(results)}")

    for i, result in enumerate(results):
        print(f"\nFile {i+1}: {result['filename']}")
        print(f"  Added lines: {len(result['added_lines'])}")
        for line in result['added_lines']:
            print(f"    {line}")
        print(f"  Removed lines: {len(result['removed_lines'])}")
        for line in result['removed_lines']:
            print(f"    {line}")

        print(f"  Is comments only: {check_add_comments_only(result)}")


def parse_hunk_line_numbers(patch_text: str) -> list[dict]:
    """
    Parse a git patch and extract the line numbers where changes were made.

    Returns a list of dicts with:
        - filename: The modified file path
        - added_line_numbers: List of line numbers where content was added
    """
    lines = patch_text.split('\n')
    files = []
    current_file = None
    added_line_numbers = []

    # Pattern to match hunk header: @@ -start,count +start,count @@
    hunk_pattern = r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@'
    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    current_line = 0

    for line in lines:
        # New file
        if line.startswith('diff --git'):
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_line_numbers': added_line_numbers
                })
            current_file = None
            added_line_numbers = []

            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)

        # Hunk header
        elif line.startswith('@@'):
            match = re.match(hunk_pattern, line)
            if match:
                current_line = int(match.group(1))

        # Added line
        elif line.startswith('+') and not line.startswith('+++'):
            added_line_numbers.append(current_line)
            current_line += 1

        # Context line (unchanged)
        elif not line.startswith('-') and not line.startswith('---') and current_file is not None:
            if line.startswith(' ') or (line and not line.startswith(('diff', '@@', 'index', '---', '+++'))):
                current_line += 1

    # Don't forget the last file
    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_line_numbers': added_line_numbers
        })

    return files


def check_lines_in_function_range(
    added_line_numbers: list[int],
    module_line_start: int,
    module_line_end: int
) -> bool:
    """
    Check if all added line numbers are within the function/class range.

    Args:
        added_line_numbers: List of line numbers where content was added
        module_line_start: Start line of the target function/class (0-indexed in data)
        module_line_end: End line of the target function/class (0-indexed in data)

    Returns:
        bool: True if all added lines are within range, False otherwise
    """
    if not added_line_numbers:
        return False

    # Convert from 0-indexed (data) to 1-indexed (git diff)
    start = module_line_start + 1
    end = module_line_end + 1

    for line_num in added_line_numbers:
        if not (start <= line_num <= end):
            return False
    return True


def evaluate_instance(instance_data: dict, output_data: dict) -> dict:
    """
    Evaluate a single instance.

    The evaluation checks:
    1. target_docstring_edited: Whether the target function/class has a docstring added
       within its line range (the primary goal).
    2. comments_only: Whether ALL changes across ALL files are comments/docstrings only.
       This allows the agent to add docstrings to other similar functions as long as
       no code is modified.

    Success requires both conditions to be met.

    Args:
        instance_data: The original instance data with target function info
        output_data: The output data with git_patch

    Returns:
        dict with evaluation results
    """
    # Count steps from history
    history = output_data.get('history') or []
    num_steps = len(history)

    result = {
        'instance_id': output_data.get('instance_id'),
        'num_steps': num_steps,
        'target_docstring_edited': False,
        'comments_only': False,
        'success': False,
    }

    git_patch = output_data.get('test_result', {}).get('git_patch', '')
    if not git_patch:
        return result

    # Parse the patch
    patch_dicts = parse_git_patch(git_patch)
    line_info = parse_hunk_line_numbers(git_patch)

    # Get expected file path(s) and line ranges
    # Support both single function and multi-function instances
    if 'functions' in instance_data and instance_data['functions']:
        targets = instance_data['functions']
    else:
        targets = [{
            'file_path': instance_data.get('file_path'),
            'module_line_start': instance_data.get('module_line_start'),
            'module_line_end': instance_data.get('module_line_end'),
        }]

    # Check if ALL added lines across ALL files are comments/docstrings only
    # This is the key change: we allow modifying multiple functions, as long as
    # all changes are comments/docstrings (no code modifications)
    all_comments = True
    for patch_dict in patch_dicts:
        if not check_add_comments_only(patch_dict):
            all_comments = False
            break
    result['comments_only'] = all_comments

    # Check if the TARGET function/class has a docstring added within its range
    # We check if ANY added lines fall within the target's line range
    # This allows the agent to also modify other functions (as long as comments_only passes)
    target_edited = False
    for file_info in line_info:
        filename = file_info['filename']
        added_lines = file_info['added_line_numbers']

        # Find matching target
        for target in targets:
            if target['file_path'] == filename:
                # Check if ANY of the added lines are within the target range
                start = target['module_line_start'] + 1  # Convert to 1-indexed
                end = target['module_line_end'] + 1
                for line_num in added_lines:
                    if start <= line_num <= end:
                        target_edited = True
                        break
            if target_edited:
                break
        if target_edited:
            break

    result['target_docstring_edited'] = target_edited

    # Overall success: target was edited + all changes are comments only
    result['success'] = result['target_docstring_edited'] and result['comments_only']

    return result


def update_output_with_eval(output_file: str, instance_id2data: dict, backup: bool = True) -> str:
    """
    Update the output.jsonl file with evaluation results.

    Adds to each entry:
    - history_length: Number of steps in the agent's history
    - eval_result: Dict with success, target_docstring_edited, comments_only

    Args:
        output_file: Path to output.jsonl from inference
        instance_id2data: Dict mapping instance_id to ground truth data
        backup: If True, create a backup of the original file

    Returns:
        Path to the updated output file
    """
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
            import shutil
            shutil.copy(output_file, backup_file)
            print(f"Created backup: {backup_file}")

    # Update each entry with eval results
    updated_entries = []
    for entry in tqdm(entries, desc='Updating output with eval results'):
        instance_id = entry.get('instance_id')

        # Calculate history length
        history = entry.get('history') or []
        history_length = len(history)

        # Get eval result
        if instance_id in instance_id2data:
            instance_data = instance_id2data[instance_id]
            eval_result = evaluate_instance(instance_data, entry)
        else:
            print(f"Warning: instance_id {instance_id} not found in data file")
            eval_result = {
                'instance_id': instance_id,
                'num_steps': history_length,
                'target_docstring_edited': False,
                'comments_only': False,
                'success': False,
            }

        # Add new fields to entry
        entry['history_length'] = history_length
        entry['eval_result'] = {
            'success': eval_result['success'],
            'target_docstring_edited': eval_result['target_docstring_edited'],
            'comments_only': eval_result['comments_only'],
        }

        updated_entries.append(entry)

    # Write back to file
    with open(output_file, 'w') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Updated {len(updated_entries)} entries in {output_file}")
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate swe_doc_gen_locate outputs')
    parser.add_argument('--output-file', required=True, help='Path to output.jsonl from inference')
    parser.add_argument('--data-file', required=True, help='HuggingFace dataset name or local JSONL file path with ground truth')
    parser.add_argument('--split', default='train', help='HuggingFace dataset split (default: train)')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results to eval_results.jsonl')
    parser.add_argument('--update-output', action='store_true', help='Update output.jsonl with history_length and eval_result fields')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup when using --update-output')
    args = parser.parse_args()

    # Load ground truth data
    instance_id2data = {}
    if args.data_file.endswith('.jsonl') or args.data_file.endswith('.json'):
        with open(args.data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    instance_id2data[data['instance_id']] = data
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.data_file, split=args.split)
        for row in dataset:
            instance_id2data[row['instance_id']] = dict(row)
    print(f"Loaded {len(instance_id2data)} instances from data file")

    # Evaluate outputs
    results = []
    correct_count = 0
    total_count = 0

    with open(args.output_file, 'r') as f:
        for line in tqdm(f, desc='Evaluating'):
            line = line.strip()
            if not line:
                continue

            output_data = json.loads(line)
            instance_id = output_data.get('instance_id')

            if instance_id not in instance_id2data:
                print(f"Warning: instance_id {instance_id} not found in data file")
                continue

            instance_data = instance_id2data[instance_id]
            eval_result = evaluate_instance(instance_data, output_data)
            results.append(eval_result)

            if eval_result['success']:
                correct_count += 1
            total_count += 1

    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Total instances: {total_count}")
    print(f"Success rate: {correct_count}/{total_count} ({100*correct_count/total_count:.2f}%)")

    # Detailed breakdown
    target_edited_count = sum(1 for r in results if r['target_docstring_edited'])
    comments_only_count = sum(1 for r in results if r['comments_only'])

    print(f"\nBreakdown:")
    print(f"  Target docstring edited: {target_edited_count}/{total_count} ({100*target_edited_count/total_count:.2f}%)")
    print(f"  Comments only (all changes): {comments_only_count}/{total_count} ({100*comments_only_count/total_count:.2f}%)")

    # Step statistics
    successful_steps = [r['num_steps'] for r in results if r['success']]
    unsuccessful_steps = [r['num_steps'] for r in results if not r['success']]

    print(f"\nStep Statistics:")
    if successful_steps:
        print(f"  Successful cases ({len(successful_steps)}):")
        print(f"    Avg steps: {sum(successful_steps)/len(successful_steps):.2f}")
        print(f"    Min steps: {min(successful_steps)}")
        print(f"    Max steps: {max(successful_steps)}")
    if unsuccessful_steps:
        print(f"  Unsuccessful cases ({len(unsuccessful_steps)}):")
        print(f"    Avg steps: {sum(unsuccessful_steps)/len(unsuccessful_steps):.2f}")
        print(f"    Min steps: {min(unsuccessful_steps)}")
        print(f"    Max steps: {max(unsuccessful_steps)}")

    # Save results
    if args.save_results:
        # Get the directory of the output file and save results there
        output_dir = os.path.dirname(args.output_file)
        results_file = os.path.join(output_dir, 'eval_results.jsonl')
        with open(results_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"\nDetailed results saved to: {results_file}")

    # Update output.jsonl with eval results
    if args.update_output:
        print("\n=== Updating output.jsonl with eval results ===")
        update_output_with_eval(
            args.output_file,
            instance_id2data,
            backup=not args.no_backup
        )
