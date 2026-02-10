"""Build the docstring-locate dataset from SWE-Gym raw instances.

The script augments each instance with the target function/class metadata and
adds a short natural-language description (generated via LLM) that will be used
at inference time to locate the definition.

Features:
- Incremental saving: Saves progress after each function is processed
- Resumable: Skips already-processed functions based on unique identifiers
- Parallelization: Supports parallel processing of repos/functions
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Sequence, Tuple
import threading

# Suppress SyntaxWarnings from AST parsing old Python code with deprecated escape sequences
warnings.filterwarnings("ignore", category=SyntaxWarning)

from datasets import load_dataset
from tqdm import tqdm

HOME_DIR = os.environ.get("HOME", "")
DOWNLOAD_DIR = os.path.join(HOME_DIR, "tmp/")
CODE_DIR = os.path.join(HOME_DIR, "OpenHands2/")

# Thread-safe lock for file writes
_write_lock = threading.Lock()


def get_function_unique_id(repo: str, file_path: str, module_name: str, line_start: int) -> str:
    """Generate a unique identifier for a function based on repo, file, name, and line."""
    key = f"{repo}::{file_path}::{module_name}::{line_start}"
    return hashlib.md5(key.encode()).hexdigest()


def load_processed_ids(output_path: str) -> set:
    """Load already-processed function IDs from existing output file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        # Reconstruct the unique ID from saved data
                        func_id = get_function_unique_id(
                            entry.get("repo", ""),
                            entry.get("file_path", ""),
                            entry.get("module_name", ""),
                            entry.get("module_line_start", 0),
                        )
                        processed_ids.add(func_id)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[warn] Error loading existing data: {e}. Starting fresh.")
    return processed_ids


def append_to_jsonl(output_path: str, entry: dict) -> None:
    """Thread-safe append a single entry to JSONL file."""
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def jsonl_to_json(jsonl_path: str, json_path: str) -> list[dict]:
    """Convert JSONL file to JSON array file and return the data."""
    data = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def clone_and_checkout(repo_name: str, commit: str, repo_dir: str) -> None:
    """Clone repo if needed and checkout the given commit."""
    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", repo_dir],
            check=True,
        )
    subprocess.run(["git", "-C", repo_dir, "checkout", commit], check=True)


def remove_repo_dir(repo_dir: str) -> None:
    """Remove cloned repository directory to save disk space."""
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir], check=True)
        print(f"[info] Removed repo dir: {repo_dir}")


def parse_diff_get_filepath_and_line(diff_text: str) -> tuple[Optional[str], List[int]]:
    """Extract a single file path and the start lines for added hunks from a diff."""
    match = re.search(r"diff --git a/(.*?) b/\1", diff_text)
    if not match:
        return None, []
    file_path = match.group(1)
    line_nums = [int(m) for m in re.findall(r"\@\@ -\d+(?:,\d+)? \+(\d+)", diff_text)]
    return file_path, line_nums


def get_all_func_w_docstrings(repo_dir: str) -> list[tuple[str, int]]:
    """Return (file_path, line_no) pairs for every function/class that has a docstring."""
    results: list[tuple[str, int]] = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if ast.get_docstring(node):
                            results.append((file_path, node.lineno))
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError, OSError):
                # Skip files that can't be parsed or don't exist (broken symlinks, etc.)
                continue
    return results


def get_node_docstring_info(filepath: str, line_nums: Sequence[int]):
    """Find the function/class that overlaps with the patch and return docstring info."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"error parsing {filepath}: {exc}")
        return None, None, None, None, None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_type = "class" if isinstance(node, ast.ClassDef) else "function"
        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        module_start_line = node.lineno - 1
        module_end_line = getattr(node, "end_lineno", module_start_line) - 1
        doc_node = node.body[0]
        if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant):
            doc_start = doc_node.lineno - 1
            doc_end = getattr(doc_node, "end_lineno", doc_start) - 1
        else:
            continue

        for line in line_nums:
            if module_start_line <= line <= module_end_line:
                return (
                    node.name,
                    node_type,
                    docstring,
                    (module_start_line, module_end_line),
                    (doc_start, doc_end),
                )
    return None, None, None, None, None


def generate_function_description(
    docstring: str,
    module_name: str,
    module_type: str,
    model: str,
    temperature: float,
) -> str:
    """Use an LLM to produce a concise, high-level description of the symbol."""
    if not docstring:
        return ""

    prompt = (
        "You are preparing a docstring-location dataset.\n"
        "Given the existing docstring for a Python function or class, write a concise\n"
        "1-3 sentence description of its behavior and purpose. Avoid quoting the\n"
        "docstring verbatim; summarize in your own words so the description can be\n"
        "used as a hint to locate the definition in the repository."
        f"\n\nName: {module_name} ({module_type})\nDocstring:\n{docstring}\n"
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        # Fall back to the first sentence of the docstring if LLM call fails.
        print(f"[warn] Falling back to raw docstring summary ({exc})")
        first_sentence = docstring.strip().split("\n")[0]
        return first_sentence


def remove_duplicate(dataset: list[dict]) -> list[dict]:
    clean_dataset = []
    existing_instance_id = set()
    for instance in dataset:
        if instance["instance_id"] in existing_instance_id:
            continue
        existing_instance_id.add(instance["instance_id"])
        clean_dataset.append(instance)
    return clean_dataset


def sample_dataset(dataset, sample_size: Optional[int], seed: int):
    if not sample_size or sample_size <= 0:
        return dataset
    sample_size = min(sample_size, len(dataset))
    if isinstance(dataset, list):
        rng = random.Random(seed)
        rng.shuffle(dataset)
        return dataset[:sample_size]
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[:sample_size])


def form_groups(
    functions: list[dict],
    min_group_size: int,
    max_group_size: int,
    max_instances: int,
    seed: int,
) -> list[list[dict]]:
    """Create groups of functions (prefer different files) for one repo."""
    rng = random.Random(seed)
    rng.shuffle(functions)
    groups: list[list[dict]] = []
    idx = 0

    while idx < len(functions):
        if max_instances and len(groups) >= max_instances:
            break
        group: list[dict] = []
        seen_files = set()
        duplicates: list[dict] = []
        group_size = rng.randint(min_group_size, max_group_size)
        while idx < len(functions) and len(group) < group_size:
            func = functions[idx]
            idx += 1
            if func["file_path"] in seen_files:
                duplicates.append(func)
                continue
            group.append(func)
            seen_files.add(func["file_path"])
        while duplicates and len(group) < group_size:
            group.append(duplicates.pop())
        if len(group) >= min_group_size:
            groups.append(group)
    return groups


def build_instance_with_description(
    instance: dict,
    module_name: str,
    module_type: str,
    file_path: str,
    module_line_range: tuple[int, int],
    docstring: str,
    docstring_line_range: tuple[int, int],
    model: str,
    temperature: float,
) -> dict:
    description = generate_function_description(
        docstring=docstring,
        module_name=module_name,
        module_type=module_type,
        model=model,
        temperature=temperature,
    )

    new_instance = instance.copy()
    new_instance["file_path"] = file_path
    new_instance["module_name"] = module_name
    new_instance["module_type"] = module_type
    new_instance["module_line_start"] = module_line_range[0]
    new_instance["module_line_end"] = module_line_range[1]
    new_instance["docstring"] = docstring
    new_instance["docstring_line_start"] = docstring_line_range[0]
    new_instance["docstring_line_end"] = docstring_line_range[1]
    new_instance["function_description"] = description
    return new_instance


def process_single_function(
    instance: dict,
    repo_dir: str,
    full_path: str,
    line_num: int,
    module_idx: int,
    model: str,
    temperature: float,
    processed_ids: set,
    output_path: str,
) -> Optional[dict]:
    """Process a single function and save incrementally. Returns the new instance or None."""
    if not os.path.exists(full_path):
        return None

    (
        module_name,
        module_type,
        docstring,
        module_line_range,
        docstring_line_range,
    ) = get_node_docstring_info(full_path, [line_num])

    if module_name is None:
        return None

    file_path = full_path.split(repo_dir + "/")[1]

    # Check if already processed using unique ID
    func_id = get_function_unique_id(
        instance["repo"],
        file_path,
        module_name,
        module_line_range[0],
    )

    if func_id in processed_ids:
        return None  # Already processed, skip

    new_instance = build_instance_with_description(
        instance=instance,
        module_name=module_name,
        module_type=module_type,
        file_path=file_path,
        module_line_range=module_line_range,
        docstring=docstring,
        docstring_line_range=docstring_line_range,
        model=model,
        temperature=temperature,
    )
    new_instance["instance_id"] = f"{instance['instance_id']}_{module_idx}"
    new_instance["image_instance_id"] = instance["instance_id"]

    # Save incrementally to JSONL file
    append_to_jsonl(output_path, new_instance)

    return new_instance


def build_all_func_dataset(
    dataset,
    dataset_save_name: str,
    split: str,
    model: str,
    temperature: float,
    sample_size: Optional[int],
    num_workers: int = 1,
) -> list[dict]:
    """Build the all_func dataset with incremental saving and resume support.

    Args:
        dataset: The source dataset
        dataset_save_name: Name for saving the dataset
        split: Dataset split (e.g., 'train')
        model: OpenAI model name
        temperature: Sampling temperature
        sample_size: Optional cap on instances
        num_workers: Number of parallel workers for processing functions

    Returns:
        List of all processed instances
    """
    # Use JSONL for incremental saves, JSON for final output
    jsonl_output_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/hybrid_gym_func_localize_all_func_{dataset_save_name}_{split}.jsonl",
    )
    json_output_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/hybrid_gym_func_localize_all_func_{dataset_save_name}_{split}.json",
    )

    # Load already processed function IDs for resume
    processed_ids = load_processed_ids(jsonl_output_path)
    print(f"[info] Found {len(processed_ids)} already processed functions. Resuming...")

    selected_instances = []
    repos = []
    for instance in tqdm(dataset, total=len(dataset), desc="Selecting repos"):
        if instance["repo"] not in repos:
            selected_instances.append(instance)
            repos.append(instance["repo"])

    selected_instances = sample_dataset(selected_instances, sample_size, seed=42)
    stat_by_repo: dict[str, dict[str, int]] = {}
    total_new_instances = 0

    for instance in tqdm(selected_instances, total=len(selected_instances), desc="Repos"):
        repo_name = instance["repo"]
        repo_dir = os.path.join(DOWNLOAD_DIR, repo_name.split("/")[-1])

        try:
            clone_and_checkout(repo_name, instance["base_commit"], repo_dir)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Skip repo {repo_name} due to checkout failure: {exc}")
            continue

        loc_pairs = get_all_func_w_docstrings(repo_dir)

        new_instance_count = 0
        file_not_found_count = 0
        func_class_not_found_count = 0
        skipped_count = 0

        if num_workers > 1:
            # Parallel processing within repo
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for module_idx, (full_path, line_num) in enumerate(loc_pairs):
                    future = executor.submit(
                        process_single_function,
                        instance,
                        repo_dir,
                        full_path,
                        line_num,
                        module_idx,
                        model,
                        temperature,
                        processed_ids,
                        jsonl_output_path,
                    )
                    futures[future] = (full_path, line_num)

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{repo_name}",
                    leave=False,
                ):
                    result = future.result()
                    if result is not None:
                        new_instance_count += 1
                        # Add to processed_ids to prevent re-processing
                        func_id = get_function_unique_id(
                            result["repo"],
                            result["file_path"],
                            result["module_name"],
                            result["module_line_start"],
                        )
                        processed_ids.add(func_id)
        else:
            # Sequential processing (original behavior)
            for module_idx, (full_path, line_num) in tqdm(
                enumerate(loc_pairs),
                total=len(loc_pairs),
                desc=f"{repo_name}",
                leave=False,
            ):
                if not os.path.exists(full_path):
                    file_not_found_count += 1
                    continue

                (
                    module_name,
                    module_type,
                    docstring,
                    module_line_range,
                    docstring_line_range,
                ) = get_node_docstring_info(full_path, [line_num])

                if module_name is None:
                    func_class_not_found_count += 1
                    continue

                file_path = full_path.split(repo_dir + "/")[1]

                # Check if already processed
                func_id = get_function_unique_id(
                    instance["repo"],
                    file_path,
                    module_name,
                    module_line_range[0],
                )

                if func_id in processed_ids:
                    skipped_count += 1
                    continue

                new_instance = build_instance_with_description(
                    instance=instance,
                    module_name=module_name,
                    module_type=module_type,
                    file_path=file_path,
                    module_line_range=module_line_range,
                    docstring=docstring,
                    docstring_line_range=docstring_line_range,
                    model=model,
                    temperature=temperature,
                )
                new_instance["instance_id"] = f"{instance['instance_id']}_{module_idx}"
                new_instance["image_instance_id"] = instance["instance_id"]

                # Save incrementally
                append_to_jsonl(jsonl_output_path, new_instance)
                processed_ids.add(func_id)
                new_instance_count += 1

        total_new_instances += new_instance_count
        stat_by_repo[repo_name] = {
            "file_not_found": file_not_found_count,
            "func_class_not_found": func_class_not_found_count,
            "new_instance_count": new_instance_count,
            "skipped_already_processed": skipped_count,
        }

        # Clean up: remove cloned repo to save disk space
        remove_repo_dir(repo_dir)

    print(json.dumps(stat_by_repo, indent=4))
    print(f"[info] Added {total_new_instances} new instances this run.")

    # Convert JSONL to final JSON format
    print(f"[info] Converting JSONL to JSON: {json_output_path}")
    new_dataset = jsonl_to_json(jsonl_output_path, json_output_path)
    print(f"[info] Total instances in dataset: {len(new_dataset)}")

    return new_dataset


def build_multi_func_dataset(
    all_func_dataset: list[dict],
    dataset_save_name: str,
    split: str,
    min_group_size: int,
    max_group_size: int,
    max_instances_per_repo: int,
    max_repos: int,
    max_functions_per_repo: Optional[int],
    seed: int,
) -> list[dict]:
    """Bundle 2–3 functions (prefer different files) into one instance."""
    existing_dataset_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/hybrid_gym_func_localize_multi_func_{dataset_save_name}_{split}.json",
    )

    # Group all-function entries by repo
    repo_map: dict[str, list[dict]] = {}
    for entry in all_func_dataset:
        repo_map.setdefault(entry["repo"], []).append(entry)

    repo_items = list(repo_map.items())
    rng = random.Random(seed)
    rng.shuffle(repo_items)
    if max_repos:
        repo_items = repo_items[:max_repos]

    new_dataset: list[dict] = []
    for repo_name, entries in repo_items:
        base_instance = entries[0]
        functions = entries.copy()
        rng.shuffle(functions)
        if max_functions_per_repo:
            functions = functions[:max_functions_per_repo]

        groups = form_groups(
            functions,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            max_instances=max_instances_per_repo,
            seed=seed,
        )

        for idx, group in enumerate(groups):
            new_instance = base_instance.copy()
            new_instance["instance_id"] = f"{base_instance['instance_id']}_multi_{idx}"
            new_instance["image_instance_id"] = base_instance["instance_id"]
            new_instance["functions"] = [
                {
                    "module_name": func["module_name"],
                    "module_type": func["module_type"],
                    "file_path": func["file_path"],
                    "module_line_start": func["module_line_start"],
                    "module_line_end": func["module_line_end"],
                    "docstring": func.get("docstring"),
                    "docstring_line_start": func["docstring_line_start"],
                    "docstring_line_end": func["docstring_line_end"],
                    "function_description": func.get("function_description"),
                }
                for func in group
            ]
            new_instance["n_functions"] = len(group)
            new_dataset.append(new_instance)

    with open(existing_dataset_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f)

    print(f"saving the new dataset to {existing_dataset_path}")
    print(f"total multi-function instances: {len(new_dataset)}")
    return new_dataset


def build_issue_func_dataset(
    dataset,
    dataset_save_name: str,
    split: str,
    model: str,
    temperature: float,
    sample_size: Optional[int],
) -> list[dict]:
    existing_dataset_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/hybrid_gym_func_localize_{dataset_save_name}_{split}.json",
    )

    new_dataset: list[dict] = []
    if os.path.exists(existing_dataset_path):
        with open(existing_dataset_path, "r", encoding="utf-8") as f:
            new_dataset = remove_duplicate(json.load(f))

    dataset = sample_dataset(dataset, sample_size, seed=42)

    file_not_found_count = 0
    func_class_not_found_count = 0
    for instance in tqdm(dataset, total=len(dataset), desc="Instances"):
        repo_name = instance["repo"]
        repo_dir = os.path.join(DOWNLOAD_DIR, repo_name.split("/")[-1])

        try:
            clone_and_checkout(repo_name, instance["base_commit"], repo_dir)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Skip repo {repo_name} due to checkout failure: {exc}")
            continue
        file_path, line_nums = parse_diff_get_filepath_and_line(instance["patch"])
        if not file_path:
            func_class_not_found_count += 1
            continue

        full_path = os.path.join(repo_dir, file_path)
        if not os.path.exists(full_path):
            file_not_found_count += 1
            continue

        (
            module_name,
            module_type,
            docstring,
            module_line_range,
            docstring_line_range,
        ) = get_node_docstring_info(full_path, line_nums)
        if module_name is None:
            func_class_not_found_count += 1
            continue

        new_instance = build_instance_with_description(
            instance=instance,
            module_name=module_name,
            module_type=module_type,
            file_path=file_path,
            module_line_range=module_line_range,
            docstring=docstring,
            docstring_line_range=docstring_line_range,
            model=model,
            temperature=temperature,
        )
        new_dataset.append(new_instance)

    with open(existing_dataset_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f)

    print(f"saving the new dataset to {existing_dataset_path}")
    print(f"file not found: {file_not_found_count}")
    print(f"func/class not found: {func_class_not_found_count}")
    return new_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SWE-Gym/SWE-Gym-Raw",
        help="Base dataset to build from (default: SWE-Gym/SWE-Gym-Raw).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["all_func", "issue_func", "multi_func"],
        default="issue_func",
        help="Whether to gather all functions, patch-overlap functions, or grouped multi-function instances.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional cap on number of instances processed (0 = all).",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Model used to synthesize function descriptions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for description generation.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum number of functions per grouped instance (multi_func mode).",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=3,
        help="Maximum number of functions per grouped instance (multi_func mode).",
    )
    parser.add_argument(
        "--max-instances-per-repo",
        type=int,
        default=50,
        help="Optional cap per repo for grouped instances (0 = no cap).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=0,
        help="Optional limit on number of repos to process for grouping (0 = all).",
    )
    parser.add_argument(
        "--max-functions-per-repo",
        type=int,
        default=0,
        help="Optional cap on functions considered per repo before grouping (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and grouping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing functions (default: 1, sequential).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset_save_name = args.dataset_name.split("/")[-1]

    if args.mode == "all_func":
        build_all_func_dataset(
            dataset,
            dataset_save_name,
            args.split,
            model=args.openai_model,
            temperature=args.temperature,
            sample_size=args.sample_size,
            num_workers=args.num_workers,
        )
    elif args.mode == "multi_func":
        all_func_dataset = build_all_func_dataset(
            dataset,
            dataset_save_name,
            args.split,
            model=args.openai_model,
            temperature=args.temperature,
            sample_size=args.sample_size,
            num_workers=args.num_workers,
        )
        build_multi_func_dataset(
            all_func_dataset=all_func_dataset,
            dataset_save_name=dataset_save_name,
            split=args.split,
            min_group_size=args.min_group_size,
            max_group_size=args.max_group_size,
            max_instances_per_repo=args.max_instances_per_repo,
            max_repos=args.max_repos,
            max_functions_per_repo=args.max_functions_per_repo
            if args.max_functions_per_repo > 0
            else None,
            seed=args.seed,
        )
    else:
        build_issue_func_dataset(
            dataset,
            dataset_save_name,
            args.split,
            model=args.openai_model,
            temperature=args.temperature,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
