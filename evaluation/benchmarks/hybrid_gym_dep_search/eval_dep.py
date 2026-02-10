"""Evaluate swe_bench_dep outputs.

This script evaluates the agent's output by:
1. Parsing git patch to find added lines
2. Checking if comments are placed in correct module ranges
3. Checking if added content is comment-only
4. Computing partial credit (how many dependencies were correctly annotated)
5. Detecting false positives (comments for non-dependencies)
6. Detecting duplicates (same dependency annotated multiple times)
7. Detecting misplaced comments (wrong location)

Metrics computed:
- True Positives (TP): Correctly annotated dependencies
- False Negatives (FN): Dependencies that were not annotated
- False Positives (FP): Comments added for things that are not dependencies
- Duplicates: Same dependency annotated multiple times
- Misplaced: Comments that mention the function but are in wrong locations
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (Precision * Recall) / (Precision + Recall)

Usage:
    python3 eval_dep.py \
        --output-file outputs/.../output.jsonl \
        --data-file resource/swe_bench_dep_data.jsonl \
        --save-results \
        --update-output
"""

import argparse
import json
import os
import re
from collections import defaultdict

from tqdm import tqdm


def parse_git_patch(patch_text: str) -> list[dict]:
    """Parse a git patch and return added lines per file.

    Returns:
        list of dicts with: filename, added_lines (list of line content strings)
    """
    if not patch_text:
        return []

    lines = patch_text.split('\n')
    files = []
    current_file = None
    added_lines = []

    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    for line in lines:
        if line.startswith('diff --git'):
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_lines': added_lines,
                })
            current_file = None
            added_lines = []

            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)

        elif line.startswith('+') and not line.startswith('+++') and current_file is not None:
            content = line[1:]  # Remove the leading '+'
            added_lines.append(content)

    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_lines': added_lines,
        })

    return files


def parse_hunk_line_numbers(patch_text: str) -> list[dict]:
    """Parse a git patch and extract line numbers where changes were made.

    Returns:
        list of dicts with: filename, added_line_numbers (list of int, 1-indexed)
    """
    if not patch_text:
        return []

    lines = patch_text.split('\n')
    files = []
    current_file = None
    added_line_numbers = []

    hunk_pattern = r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@'
    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    current_line = 0

    for line in lines:
        if line.startswith('diff --git'):
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_line_numbers': added_line_numbers
                })
            current_file = None
            added_line_numbers = []
            current_line = 0

            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)

        elif line.startswith('@@'):
            match = re.match(hunk_pattern, line)
            if match:
                current_line = int(match.group(1))

        elif line.startswith('+') and not line.startswith('+++'):
            if current_file is not None:
                added_line_numbers.append(current_line)
            current_line += 1

        elif line.startswith('-') and not line.startswith('---'):
            # Deleted lines don't advance the new file line counter
            pass

        elif current_file is not None and not line.startswith(('diff', '@@', 'index', '---', '+++')):
            # Context line (or empty line in diff)
            current_line += 1

    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_line_numbers': added_line_numbers
        })

    return files


def parse_patch_with_details(patch_text: str) -> list[dict]:
    """Parse a git patch and extract detailed info about added lines.

    Returns:
        list of dicts with:
            - filename
            - added_lines (list of dicts with line_number and content)
            - line_mapping: function to convert original line -> new line
    """
    if not patch_text:
        return []

    lines = patch_text.split('\n')
    files = []
    current_file = None
    added_lines = []
    # Track (old_start, old_count, new_start, new_count) for each hunk
    hunks = []

    hunk_pattern = r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    current_line = 0

    for line in lines:
        if line.startswith('diff --git'):
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_lines': added_lines,
                    'hunks': hunks,
                })
            current_file = None
            added_lines = []
            hunks = []
            current_line = 0

            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)

        elif line.startswith('@@'):
            match = re.match(hunk_pattern, line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                hunks.append((old_start, old_count, new_start, new_count))
                current_line = new_start

        elif line.startswith('+') and not line.startswith('+++'):
            if current_file is not None:
                content = line[1:]  # Remove leading '+'
                added_lines.append({
                    'line_number': current_line,
                    'content': content
                })
            current_line += 1

        elif line.startswith('-') and not line.startswith('---'):
            # Deleted lines don't advance the new file line counter
            pass

        elif current_file is not None and not line.startswith(('diff', '@@', 'index', '---', '+++')):
            # Context line
            current_line += 1

    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_lines': added_lines,
            'hunks': hunks,
        })

    return files


def compute_line_offset(hunks: list, original_line: int) -> int:
    """Compute the line offset for an original line number based on patch hunks.

    Given hunks from a patch, compute how much a line in the original file
    has shifted in the new file due to additions/deletions.

    Args:
        hunks: List of (old_start, old_count, new_start, new_count) tuples
        original_line: Line number in the original file (1-indexed)

    Returns:
        The offset to add to original_line to get the new file line number.
        Returns 0 if the line is before all hunks.
    """
    if not hunks:
        return 0

    cumulative_offset = 0
    for old_start, old_count, new_start, new_count in sorted(hunks, key=lambda h: h[0]):
        # If original_line is before this hunk, use the current cumulative offset
        if original_line < old_start:
            return cumulative_offset
        # If original_line is within this hunk's range in the original file
        old_end = old_start + old_count
        if original_line < old_end:
            # Line is within the hunk. When adding comments before a function,
            # the additions typically come BEFORE the original content (context lines).
            # So original lines are pushed down by the number of added lines.
            # Example: @@ -10,1 +10,5 @@ adds 4 lines before line 10,
            # pushing original line 10 to new line 14 (offset = 5-1 = 4)
            lines_added_or_removed = new_count - old_count
            return cumulative_offset + lines_added_or_removed
        # Line is after this hunk, accumulate the offset
        cumulative_offset = (new_start + new_count) - (old_start + old_count)

    return cumulative_offset


def is_line_comment_only(line: str) -> bool:
    """Check if a single line is a comment only."""
    stripped = line.strip()
    if not stripped:
        return True  # Empty lines are fine
    return stripped.startswith('#')


def is_lines_comment_only(lines: list[str]) -> bool:
    """Check if all lines are comments only (no code changes)."""
    for line in lines:
        if not is_line_comment_only(line):
            return False
    return True


def check_comment_content(line: str, target_func_name: str) -> bool:
    """Check if the comment mentions the target function correctly.

    Flexible matching allows for:
    - Case insensitivity
    - Optional "function/class" vs just "function" or "class"
    - Trailing punctuation (period, etc.)
    - Minor variations in phrasing
    - The comment should indicate this code is called/used by the target function
    """
    stripped = line.strip().lower()
    if not stripped.startswith('#'):
        return False

    # Remove the # and normalize content
    content = stripped[1:].strip()

    # Remove trailing punctuation
    content = content.rstrip('.,;:!')

    target_lower = target_func_name.lower()

    # Check for various acceptable patterns
    acceptable_patterns = [
        # Standard patterns
        f"this function/class is called by the {target_lower} function",
        f"this function is called by the {target_lower} function",
        f"this class is called by the {target_lower} function",
        f"this method is called by the {target_lower} function",
        # Without "the"
        f"this function/class is called by {target_lower} function",
        f"this function/class is called by {target_lower}",
        f"this function is called by {target_lower}",
        f"this class is called by {target_lower}",
        # Alternative phrasings
        f"called by the {target_lower} function",
        f"called by {target_lower}",
        f"used by the {target_lower} function",
        f"used by {target_lower}",
        f"dependency of {target_lower}",
        f"this is a dependency of {target_lower}",
        # With "method" instead of "function"
        f"this function/class is called by the {target_lower} method",
        f"this function is called by the {target_lower} method",
    ]

    # Check exact matches first
    if content in acceptable_patterns:
        return True

    # Fuzzy matching: check if the content contains key indicators
    # Must mention the target function name and indicate it's a caller/dependency relationship
    has_target = target_lower in content
    has_called_by = 'called by' in content or 'used by' in content or 'dependency' in content

    if has_target and has_called_by:
        return True

    return False


def evaluate_instance(instance_data: dict, output_data: dict) -> dict:
    """Evaluate a single instance with comprehensive metrics.

    Args:
        instance_data: Ground truth data with dependencies
        output_data: Agent output with git_patch

    Returns:
        dict with evaluation metrics including:
        - true_positives: correctly annotated dependencies
        - false_negatives: missed dependencies
        - false_positives: incorrect annotations (wrong location or non-dependencies)
        - duplicates: same dependency annotated multiple times
        - precision, recall, f1
    """
    target_func_name = instance_data['target_function_name']
    dependencies = instance_data['dependencies']
    num_deps = len(dependencies)

    result = {
        'instance_id': output_data.get('instance_id'),
        'target_function': target_func_name,
        'num_dependencies': num_deps,
        # Core metrics
        'true_positives': 0,
        'false_negatives': 0,
        'false_positives': 0,
        'duplicates': 0,
        # Derived metrics
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'partial_score': 0.0,
        'full_success': False,
        'comments_only': False,
        # Lists for details
        'annotated_deps': [],
        'missed_deps': [],
        'false_positive_comments': [],
        'duplicate_comments': [],
        'details': [],
    }

    git_patch = output_data.get('test_result', {}).get('git_patch', '')
    if not git_patch:
        result['missed_deps'] = [d['name'] for d in dependencies]
        result['false_negatives'] = num_deps
        return result

    # Parse patch with detailed line info (includes hunks for line offset calculation)
    patch_details = parse_patch_with_details(git_patch)

    # Check if all changes are comments only
    all_lines = []
    for pf in patch_details:
        all_lines.extend([l['content'] for l in pf['added_lines']])
    result['comments_only'] = is_lines_comment_only(all_lines)

    # Build a map of file -> hunks for line offset calculation
    file_hunks = {}
    for pf in patch_details:
        file_hunks[pf['filename']] = pf.get('hunks', [])

    # Build a map of expected dependency locations (in ORIGINAL file coordinates)
    # Key: (file_path, original_line_1idx) -> dep_info
    expected_deps = {}
    for dep in dependencies:
        dep_file = dep['file_path']
        expected_line_0idx = dep.get('decorator_line', dep['line_start'])
        expected_line_1idx = expected_line_0idx + 1
        expected_deps[(dep_file, expected_line_1idx)] = dep

    # Track which dependencies were found
    found_deps = set()  # Set of dep names that were correctly annotated
    dep_annotation_counts = defaultdict(int)  # Count annotations per dependency

    # Track all comments that mention the target function
    all_relevant_comments = []  # All comments that mention the target function

    # Analyze all added lines
    for file_info in patch_details:
        filename = file_info['filename']
        hunks = file_info.get('hunks', [])

        for added_line in file_info['added_lines']:
            line_num = added_line['line_number']  # Line number in NEW file
            content = added_line['content']

            # Check if this is a comment that mentions the target function
            if check_comment_content(content, target_func_name):
                comment_info = {
                    'file': filename,
                    'line': line_num,
                    'content': content.strip(),
                    'matched_dep': None,
                    'status': 'unknown',  # Will be: 'correct', 'duplicate', 'false_positive'
                }
                all_relevant_comments.append(comment_info)

                # Try to match this comment to an expected dependency
                matched = False
                for (dep_file, original_expected_line), dep in expected_deps.items():
                    if dep_file != filename:
                        continue

                    # Compute the adjusted expected line in NEW file coordinates
                    line_offset = compute_line_offset(hunks, original_expected_line)
                    adjusted_expected_line = original_expected_line + line_offset

                    # Check if line is close enough (within 5 lines tolerance)
                    if abs(line_num - adjusted_expected_line) <= 5:
                        comment_info['matched_dep'] = dep['name']
                        dep_annotation_counts[dep['name']] += 1

                        if dep['name'] in found_deps:
                            # This is a duplicate
                            comment_info['status'] = 'duplicate'
                        else:
                            # First correct annotation for this dep
                            comment_info['status'] = 'correct'
                            found_deps.add(dep['name'])
                        matched = True
                        break

                if not matched:
                    # Comment mentions target function but doesn't match any dependency
                    # This is a false positive (wrong location or wrong file)
                    comment_info['status'] = 'false_positive'

    # Calculate metrics
    true_positives = len(found_deps)
    false_negatives = num_deps - true_positives

    # Count false positives and duplicates from comments
    false_positives = sum(1 for c in all_relevant_comments if c['status'] == 'false_positive')
    duplicates = sum(1 for c in all_relevant_comments if c['status'] == 'duplicate')

    # Calculate precision/recall/f1
    # Precision: TP / (TP + FP + duplicates) - all positive predictions
    # Recall: TP / (TP + FN)
    total_positive_predictions = true_positives + false_positives + duplicates
    precision = true_positives / total_positive_predictions if total_positive_predictions > 0 else 0.0
    recall = true_positives / num_deps if num_deps > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Build result
    result['true_positives'] = true_positives
    result['false_negatives'] = false_negatives
    result['false_positives'] = false_positives
    result['duplicates'] = duplicates

    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    result['partial_score'] = recall  # Same as before (TP / total deps)

    result['annotated_deps'] = list(found_deps)
    result['missed_deps'] = [d['name'] for d in dependencies if d['name'] not in found_deps]
    result['false_positive_comments'] = [c for c in all_relevant_comments if c['status'] == 'false_positive']
    result['duplicate_comments'] = [c for c in all_relevant_comments if c['status'] == 'duplicate']

    # Full success requires: all deps annotated, no false positives, no duplicates, comments only
    result['full_success'] = (
        true_positives == num_deps and
        false_positives == 0 and
        duplicates == 0 and
        result['comments_only']
    )

    # Detailed per-dependency info
    details = []
    for dep in dependencies:
        dep_file = dep['file_path']
        dep_name = dep['name']
        expected_line_0idx = dep.get('decorator_line', dep['line_start'])
        expected_line_1idx = expected_line_0idx + 1

        dep_detail = {
            'name': dep_name,
            'file': dep_file,
            'expected_line': expected_line_1idx,
            'found': dep_name in found_deps,
            'annotation_count': dep_annotation_counts.get(dep_name, 0),
        }
        details.append(dep_detail)

    result['details'] = details

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate swe_bench_dep outputs')
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output.jsonl from inference'
    )
    parser.add_argument(
        '--data-file',
        required=True,
        help='HuggingFace dataset name or local JSONL file path with ground truth'
    )
    parser.add_argument(
        '--split',
        default='train',
        help='HuggingFace dataset split (default: train)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to eval_results.jsonl'
    )
    parser.add_argument(
        '--update-output',
        action='store_true',
        help='Update output.jsonl with eval_result field'
    )
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
    full_success_count = 0
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

            if eval_result['full_success']:
                full_success_count += 1
            total_count += 1

    # Aggregate metrics
    total_tp = sum(r['true_positives'] for r in results)
    total_fn = sum(r['false_negatives'] for r in results)
    total_fp = sum(r['false_positives'] for r in results)
    total_duplicates = sum(r['duplicates'] for r in results)
    total_deps = sum(r['num_dependencies'] for r in results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total instances evaluated: {total_count}")

    # Initialize breakdown dict (will be populated if total_count > 0)
    num_deps_breakdown = {}

    if total_count > 0:
        # Primary metrics
        print(f"\n--- Primary Metrics ---")
        print(f"Full success rate: {full_success_count}/{total_count} ({100*full_success_count/total_count:.2f}%)")

        avg_precision = sum(r['precision'] for r in results) / total_count
        avg_recall = sum(r['recall'] for r in results) / total_count
        avg_f1 = sum(r['f1'] for r in results) / total_count
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average F1:        {avg_f1:.4f}")

        # Micro-averaged metrics (aggregated across all instances)
        # Include duplicates in denominator for precision
        micro_precision = total_tp / (total_tp + total_fp + total_duplicates) if (total_tp + total_fp + total_duplicates) > 0 else 0.0
        micro_recall = total_tp / total_deps if total_deps > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        print(f"\n--- Micro-averaged Metrics ---")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall:    {micro_recall:.4f}")
        print(f"Micro F1:        {micro_f1:.4f}")

        # Confusion matrix style
        print(f"\n--- Confusion Matrix ---")
        print(f"True Positives (correct annotations):  {total_tp}")
        print(f"False Negatives (missed dependencies): {total_fn}")
        print(f"False Positives (wrong annotations):   {total_fp}")
        print(f"Duplicates (repeated annotations):     {total_duplicates}")

        # Comments only breakdown
        comments_only_count = sum(1 for r in results if r['comments_only'])
        print(f"\n--- Code Quality ---")
        print(f"Comments only (no code changes): {comments_only_count}/{total_count} ({100*comments_only_count/total_count:.2f}%)")

        # Distribution of recall (partial scores)
        score_bins = defaultdict(int)
        for r in results:
            bin_key = round(r['recall'], 1)
            score_bins[bin_key] += 1

        print(f"\n--- Recall Distribution ---")
        for score in sorted(score_bins.keys()):
            pct = 100 * score_bins[score] / total_count
            bar = '#' * int(pct / 2)
            print(f"  {score:.1f}: {score_bins[score]:4d} ({pct:5.1f}%) {bar}")

        # Dependency coverage stats
        print(f"\n--- Dependency Coverage ---")
        print(f"Total dependencies: {total_deps}")
        print(f"Total correctly annotated: {total_tp} ({100*total_tp/total_deps:.2f}%)")
        print(f"Total missed: {total_fn} ({100*total_fn/total_deps:.2f}%)")

        # Error analysis
        if total_fp > 0 or total_duplicates > 0:
            print(f"\n--- Error Analysis ---")
            instances_with_fp = sum(1 for r in results if r['false_positives'] > 0)
            instances_with_dup = sum(1 for r in results if r['duplicates'] > 0)
            print(f"Instances with false positives: {instances_with_fp}/{total_count}")
            print(f"Instances with duplicates:      {instances_with_dup}/{total_count}")

        # Results breakdown by num_dependencies
        print(f"\n--- Results by Number of Dependencies ---")
        results_by_num_deps = defaultdict(list)
        for r in results:
            results_by_num_deps[r['num_dependencies']].append(r)

        num_deps_breakdown = {}
        for num_deps in sorted(results_by_num_deps.keys()):
            group_results = results_by_num_deps[num_deps]
            group_count = len(group_results)
            group_success = sum(1 for r in group_results if r['full_success'])
            group_tp = sum(r['true_positives'] for r in group_results)
            group_fn = sum(r['false_negatives'] for r in group_results)
            group_fp = sum(r['false_positives'] for r in group_results)
            group_dup = sum(r['duplicates'] for r in group_results)
            group_total_deps = sum(r['num_dependencies'] for r in group_results)

            # Macro-averaged metrics for this group
            group_avg_precision = sum(r['precision'] for r in group_results) / group_count if group_count > 0 else 0.0
            group_avg_recall = sum(r['recall'] for r in group_results) / group_count if group_count > 0 else 0.0
            group_avg_f1 = sum(r['f1'] for r in group_results) / group_count if group_count > 0 else 0.0

            # Micro-averaged metrics for this group
            group_micro_precision = group_tp / (group_tp + group_fp + group_dup) if (group_tp + group_fp + group_dup) > 0 else 0.0
            group_micro_recall = group_tp / group_total_deps if group_total_deps > 0 else 0.0
            group_micro_f1 = 2 * group_micro_precision * group_micro_recall / (group_micro_precision + group_micro_recall) if (group_micro_precision + group_micro_recall) > 0 else 0.0

            print(f"\n  {num_deps} dependency/dependencies ({group_count} instances):")
            print(f"    Full success: {group_success}/{group_count} ({100*group_success/group_count:.1f}%)")
            print(f"    Avg Precision: {group_avg_precision:.4f}  Avg Recall: {group_avg_recall:.4f}  Avg F1: {group_avg_f1:.4f}")
            print(f"    TP: {group_tp}  FN: {group_fn}  FP: {group_fp}  Duplicates: {group_dup}")

            num_deps_breakdown[num_deps] = {
                'count': group_count,
                'full_success_count': group_success,
                'full_success_rate': group_success / group_count if group_count > 0 else 0.0,
                'avg_precision': group_avg_precision,
                'avg_recall': group_avg_recall,
                'avg_f1': group_avg_f1,
                'micro_precision': group_micro_precision,
                'micro_recall': group_micro_recall,
                'micro_f1': group_micro_f1,
                'true_positives': group_tp,
                'false_negatives': group_fn,
                'false_positives': group_fp,
                'duplicates': group_dup,
            }

    # Save detailed results
    if args.save_results:
        output_dir = os.path.dirname(args.output_file)
        results_file = os.path.join(output_dir, 'eval_results.jsonl')
        with open(results_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"\nDetailed results saved to: {results_file}")

        # Also save summary
        summary_file = os.path.join(output_dir, 'eval_summary.json')
        summary = {
            'total_instances': total_count,
            'full_success_count': full_success_count,
            'full_success_rate': full_success_count / total_count if total_count > 0 else 0,
            # Averaged metrics
            'avg_precision': avg_precision if total_count > 0 else 0,
            'avg_recall': avg_recall if total_count > 0 else 0,
            'avg_f1': avg_f1 if total_count > 0 else 0,
            # Micro-averaged metrics
            'micro_precision': micro_precision if total_count > 0 else 0,
            'micro_recall': micro_recall if total_count > 0 else 0,
            'micro_f1': micro_f1 if total_count > 0 else 0,
            # Counts
            'total_dependencies': total_deps if total_count > 0 else 0,
            'total_true_positives': total_tp if total_count > 0 else 0,
            'total_false_negatives': total_fn if total_count > 0 else 0,
            'total_false_positives': total_fp if total_count > 0 else 0,
            'total_duplicates': total_duplicates if total_count > 0 else 0,
            'comments_only_count': comments_only_count if total_count > 0 else 0,
            # Breakdown by number of dependencies
            'by_num_dependencies': num_deps_breakdown if total_count > 0 else {},
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")

    # Update output.jsonl with eval results
    if args.update_output:
        print(f"\n{'='*70}")
        print("Updating output.jsonl with eval results...")

        entries = []
        with open(args.output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        # Create result lookup
        result_lookup = {r['instance_id']: r for r in results}

        updated_count = 0
        with open(args.output_file, 'w') as f:
            for entry in entries:
                iid = entry.get('instance_id')
                if iid in result_lookup:
                    # Add eval_result field (without verbose lists)
                    eval_result = result_lookup[iid].copy()
                    # Remove verbose details to keep output compact
                    eval_result.pop('details', None)
                    eval_result.pop('false_positive_comments', None)
                    eval_result.pop('duplicate_comments', None)
                    entry['eval_result'] = eval_result
                    updated_count += 1
                f.write(json.dumps(entry) + '\n')

        print(f"Updated {updated_count} entries in {args.output_file}")


if __name__ == '__main__':
    main()
