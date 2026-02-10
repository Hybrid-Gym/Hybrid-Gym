import os
import json
import argparse
import re
from tqdm import tqdm

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
            - patch_lines: List of all lines in the patch for this file
    """
    # Split the patch into lines
    lines = patch_text.split('\n')

    files = []
    current_file = None
    added_lines = []
    removed_lines = []
    patch_lines = []
    is_new_file = False

    # Pattern to match file path in diff header
    file_pattern = r'^diff --git a/(.+) b/(.+)$'

    for line_num, line in enumerate(lines):
        # Extract filename from diff header
        if line.startswith('diff --git'):
            # Save previous file if exists
            if current_file is not None:
                files.append({
                    'filename': current_file,
                    'added_lines': added_lines,
                    'removed_lines': removed_lines,
                    'is_new_file': is_new_file,
                    'patch_lines': patch_lines,
                })

                current_file = None
                added_lines = []
                removed_lines = []
                patch_lines = []
                is_new_file = False

            # Start new file
            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)  # Use the 'b/' path (new file)
                patch_lines.append(line)
                # Check if this is a new file by looking ahead
                # We'll check in the next few lines for 'new file mode'
                is_new_file = False

        # Check for 'new file mode' indicator (appears after diff --git)
        elif current_file is not None and line.startswith('new file mode'):
            is_new_file = True
            patch_lines.append(line)

        # Collect all lines for the current file's patch
        elif current_file is not None:
            patch_lines.append(line)

            # Extract added lines (lines starting with +)
            if line.startswith('+') and not line.startswith('+++'):
                content = line[1:] if line.startswith('+') else line
                content = content.strip()
                if content != '':
                    added_lines.append(content)

            # Extract removed lines (lines starting with -)
            elif line.startswith('-') and not line.startswith('---'):
                content = line[1:] if line.startswith('-') else line
                content = content.strip()
                if content != '':
                    removed_lines.append(content)

    # Don't forget the last file
    if current_file is not None:
        files.append({
            'filename': current_file,
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'is_new_file': is_new_file,
            'patch_lines': patch_lines
        })

    return files


def check_add_comments_only_v0(patch_dict):
    """
    Check if a patch dictionary only adds comments and doesn't remove any lines.

    Args:
        patch_dict (dict): Dictionary containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)

    Returns:
        bool: True if only comments were added, False otherwise
    """

    has_removals = len(patch_dict['removed_lines']) > 0
    if has_removals:
        return False

    # Check if all added lines are comments
    non_comment_additions = []
    for line in patch_dict['added_lines']:
        line = line.strip()

        # Check if it's a comment (starts with # or is empty/whitespace)
        is_comment = (
            line.startswith('#') or
            line == '' or
            line.isspace()
        )

        if not is_comment:
            return False

    return True


def check_add_comments_only(patch_dict):
    """
    Check if a patch dictionary only adds comments and doesn't remove any lines.

    Args:
        patch_dict (dict): Dictionary containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)
            - patch_lines: List of all lines in the patch for this file

    Returns:
        bool: True if only comments were added, False otherwise
    """

    if patch_dict['is_new_file']:
        return True

    # Create a set of removed lines (stripped) for matching
    removed_lines_stripped = {line.strip() for line in patch_dict['removed_lines']}

    # Track which removed lines have been matched
    matched_removed_lines = set()

    # Check if all added lines are comments or match removed lines with comments appended
    for line_num, line in enumerate(patch_dict['added_lines']):
        line = line.strip()

        # Check if it's a pure comment (starts with # or is empty/whitespace)
        is_pure_comment = (
            line.startswith('#') or
            line == '' or
            line.isspace()
        )

        if is_pure_comment:
            continue

        # Check if it's a removed line with an inline comment appended
        # Split on '#' to separate code from comment
        if '#' in line:
            code_part = line.split('#', 1)[0].strip()
            # Check if the code part matches any removed line
            if code_part in removed_lines_stripped:
                matched_removed_lines.add(code_part)
                continue

        # Check if it exactly matches a removed line (same code, no comment added)
        if line in removed_lines_stripped:
            matched_removed_lines.add(line)
            continue

        # If it's neither a pure comment nor a removed line (with or without comment), it's not comment-only
        return False

    # Check if all removed lines are either comments or have been matched
    for line_num, line in enumerate(patch_dict['removed_lines']):
        line_stripped = line.strip()
        # If it's a comment or empty, it's fine
        if line_stripped.startswith('#') or line_stripped == '' or line_stripped.isspace():
            continue
        # Otherwise, it must have been matched by an added line
        if line_stripped not in matched_removed_lines:
            return False

    return True


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate issue localization outputs')
    parser.add_argument('--output-file', required=True, help='Path to output.jsonl from inference')
    parser.add_argument('--save-failure-ids', action="store_true", help='Save failure instance IDs to JSON')
    args = parser.parse_args()

    input_file = args.output_file
    assert os.path.exists(input_file), f"File not found: {input_file}"

    # Load gold patches from the instance field in output.jsonl
    instance_id2golden_patch = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            instance_id2golden_patch[data["instance_id"]] = data["instance"]["patch"]

    # extract all file paths from the golden patch
    instance_id2file_paths = {}
    for instance_id, golden_patch in tqdm(instance_id2golden_patch.items(), total=len(instance_id2golden_patch)):
        file_paths = patch2file_paths(golden_patch)
        instance_id2file_paths[instance_id] = file_paths
    print("finished extracting file paths")

    correct_count = 0
    total_count = 0
    tmp_count = 0
    empty_count = 0
    wrong_localization_count = 0
    wrong_comments_count = 0
    success_data = []
    failure_ids = []
    with open(input_file, "r") as f:
        for lid,line in tqdm(enumerate(f)):
            data = json.loads(line)

            try:
                instance_id = data["instance_id"]
                generated_patch = data["test_result"]["git_patch"]
                total_count += 1
            except:
                print(f"Error loading data: line {lid}")
                continue

            if generated_patch.strip() == "":
                empty_count += 1
                continue

            # Parse the patch using our function
            patch_dicts = parse_git_patch(generated_patch)

            comments_only_flag = True
            for patch_dict in patch_dicts:
                if not check_add_comments_only(patch_dict):
                # if not check_add_comments_only_v0(patch_dict):
                    comments_only_flag = False
                    break

            gt_files = instance_id2file_paths[instance_id]
            gen_files = [patch_dict['filename'] for patch_dict in patch_dicts]

            localization_flag = len(set(gt_files).intersection(set(gen_files))) > 0
            # localization_flag = len(set(gt_files).intersection(set(gen_files))) == len(gt_files) and len(gen_files) == len(gt_files)

            if localization_flag and comments_only_flag:
                correct_count += 1
                success_data.append(data)
            else:
                failure_ids.append(instance_id)
                if not localization_flag:
                    wrong_localization_count += 1
                if localization_flag and not comments_only_flag:
                    # from IPython import embed; embed();
                    # if lid > 10:
                    #     exit()
                    wrong_comments_count += 1
    print(f"Correct percentage: {correct_count / total_count:.4f} ({correct_count}/{total_count})")
    print(f"Empty percentage: {empty_count / total_count:.4f} ({empty_count}/{total_count})")
    print(f"Wrong localization percentage: {wrong_localization_count / total_count:.4f} ({wrong_localization_count}/{total_count})")
    print(f"Wrong comments percentage: {wrong_comments_count / total_count:.4f} ({wrong_comments_count}/{total_count})")

    if args.save_failure_ids:
        output_file = input_file.replace(".jsonl", "_failure_ids.json")
        with open(output_file, "w") as f:
            json.dump(failure_ids, f, indent=4)
