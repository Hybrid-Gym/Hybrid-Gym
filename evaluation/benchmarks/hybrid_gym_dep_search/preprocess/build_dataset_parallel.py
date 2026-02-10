"""Parallel version of build_dataset.py for faster dataset generation.

This script processes multiple repos in parallel using multiprocessing.

Usage:
    python3 build_dataset_parallel.py \
        --max-instances 4000 \
        --min-deps 1 \
        --max-deps 3 \
        --sample-per-repo 20 \
        --output resource/swe_bench_dep_data_full.jsonl \
        --num-workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Import from original module
from build_dataset import (
    PYTHON_BUILTINS,
    CallLocationExtractor,
    RepoAnalyzer,
    clone_and_checkout,
    find_suitable_functions,
    get_instance_id,
    remove_repo_dir,
)

from datasets import load_dataset
from tqdm import tqdm

HOME_DIR = os.environ.get("HOME", "")
BASE_DOWNLOAD_DIR = os.path.join(HOME_DIR, "tmp/swe_bench_dep_repos_parallel/")


def process_single_repo(args: tuple) -> dict:
    """Process a single repo and return results.

    This function runs in a worker process.

    Args:
        args: Tuple of (worker_id, repo_name, base_commit, min_deps, max_deps, sample_per_repo, seed)

    Returns:
        dict with 'status', 'samples', 'error' keys
    """
    worker_id, repo_name, base_commit, min_deps, max_deps, sample_per_repo, seed = args

    # Each worker uses its own directory to avoid conflicts
    worker_download_dir = os.path.join(BASE_DOWNLOAD_DIR, f"worker_{worker_id}")
    os.makedirs(worker_download_dir, exist_ok=True)

    repo_dir = os.path.join(worker_download_dir, repo_name.split('/')[-1] + "_" + base_commit[:8])

    result = {
        'status': 'error',
        'repo': repo_name,
        'commit': base_commit,
        'samples': [],
        'error': None,
    }

    try:
        # Clone and checkout
        if not clone_and_checkout(repo_name, base_commit, repo_dir):
            result['status'] = 'clone_failed'
            return result

        # Find suitable functions
        functions = find_suitable_functions(repo_dir, min_deps, max_deps)

        if not functions:
            result['status'] = 'no_functions'
            remove_repo_dir(repo_dir)
            return result

        # Use seed based on repo name for reproducibility
        repo_seed = hash(repo_name + base_commit) % (2**32)
        random.seed(repo_seed)

        # Group by dependency count
        by_dep_count = defaultdict(list)
        for func in functions:
            by_dep_count[func['num_dependencies']].append(func)

        # Shuffle each group
        for dep_list in by_dep_count.values():
            random.shuffle(dep_list)

        # Sample up to sample_per_repo functions, trying to balance
        sampled = []
        remaining = sample_per_repo
        dep_counts_list = list(range(min_deps, max_deps + 1))

        while remaining > 0:
            added = False
            for dep_count in dep_counts_list:
                if remaining <= 0:
                    break
                if by_dep_count[dep_count]:
                    func_data = by_dep_count[dep_count].pop()
                    sample = {
                        'instance_id': get_instance_id(repo_name, base_commit, len(sampled)),
                        'repo': repo_name,
                        'base_commit': base_commit,
                        **func_data
                    }
                    sampled.append(sample)
                    remaining -= 1
                    added = True
            if not added:
                break

        result['status'] = 'success'
        result['samples'] = sampled

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    finally:
        # Always clean up
        remove_repo_dir(repo_dir)

    return result


def build_dataset_parallel(
    dataset_name: str = "SWE-Gym/SWE-Gym-Raw",
    split: str = "train",
    sample_per_repo: int = 50,
    min_deps: int = 1,
    max_deps: int = 5,
    output_path: str = "resource/swe_bench_dep_data.jsonl",
    seed: int = 42,
    max_repos: int = 0,
    max_instances: int = 0,
    num_workers: int = 8,
):
    """Main function to build the dataset with parallel processing.

    Args:
        num_workers: Number of parallel worker processes
    """
    random.seed(seed)

    # Load dataset
    print(f"Loading dataset {dataset_name} split={split}...")
    dataset = load_dataset(dataset_name, split=split)

    # Get unique repos
    repos_seen = set()
    unique_instances = []
    for instance in dataset:
        if instance['repo'] not in repos_seen:
            unique_instances.append(instance)
            repos_seen.add(instance['repo'])

    print(f"Found {len(unique_instances)} unique repos")

    # Limit repos if specified
    if max_repos > 0:
        unique_instances = unique_instances[:max_repos]
        print(f"Limiting to {max_repos} repos")

    if max_instances > 0:
        print(f"Target: {max_instances} instances with balanced dep distribution")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

    # Load already processed repos
    processed_repos = set()
    processed_ids = set()
    dep_counts = {i: 0 for i in range(min_deps, max_deps + 1)}

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    processed_ids.add(entry.get('instance_id', ''))
                    # Track processed repos
                    repo_prefix = entry.get('repo', '').replace('/', '__') + "__" + entry.get('base_commit', '')[:8]
                    processed_repos.add(repo_prefix)
                    # Track dep counts
                    n = entry.get('num_dependencies', 0)
                    if min_deps <= n <= max_deps:
                        dep_counts[n] = dep_counts.get(n, 0) + 1

        if processed_ids:
            print(f"[info] Found {len(processed_ids)} already processed instances from {len(processed_repos)} repos. Resuming...")

    # Filter out already processed repos
    repos_to_process = []
    for instance in unique_instances:
        repo_name = instance['repo']
        base_commit = instance['base_commit']
        repo_prefix = repo_name.replace('/', '__') + "__" + base_commit[:8]
        if repo_prefix not in processed_repos:
            repos_to_process.append((repo_name, base_commit))

    print(f"Repos to process: {len(repos_to_process)}")
    print(f"Using {num_workers} parallel workers")

    # Prepare work items
    work_items = []
    for i, (repo_name, base_commit) in enumerate(repos_to_process):
        worker_id = i % num_workers
        work_items.append((worker_id, repo_name, base_commit, min_deps, max_deps, sample_per_repo, seed))

    # Process in parallel
    total_samples = len(processed_ids)
    stats = {
        'repos_processed': 0,
        'repos_clone_failed': 0,
        'repos_no_functions': 0,
        'repos_error': 0,
        'total_functions': 0,
    }

    # Open file for appending
    output_file = open(output_path, 'a')

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_repo, item): item for item in work_items}

            # Process results as they complete
            with tqdm(total=len(futures), desc="Processing repos") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        stats['repos_processed'] += 1

                        # Filter samples based on current dep_counts to maintain balance
                        samples_to_add = []
                        for sample in result['samples']:
                            # Check if we've reached max_instances
                            if max_instances > 0 and total_samples >= max_instances:
                                break

                            n = sample['num_dependencies']

                            # Add sample
                            samples_to_add.append(sample)
                            dep_counts[n] = dep_counts.get(n, 0) + 1
                            total_samples += 1

                        # Write samples to file
                        for sample in samples_to_add:
                            output_file.write(json.dumps(sample) + "\n")
                        output_file.flush()

                        stats['total_functions'] += len(samples_to_add)

                    elif result['status'] == 'clone_failed':
                        stats['repos_clone_failed'] += 1
                    elif result['status'] == 'no_functions':
                        stats['repos_no_functions'] += 1
                    else:
                        stats['repos_error'] += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'samples': total_samples,
                        'success': stats['repos_processed'],
                    })

                    # Early exit if we've reached max_instances
                    if max_instances > 0 and total_samples >= max_instances:
                        print(f"\n[info] Reached target of {max_instances} instances. Stopping...")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

    finally:
        output_file.close()

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"Repos processed successfully: {stats['repos_processed']}")
    print(f"Repos clone failed: {stats['repos_clone_failed']}")
    print(f"Repos with no suitable functions: {stats['repos_no_functions']}")
    print(f"Repos with errors: {stats['repos_error']}")
    print(f"Total samples created: {total_samples}")
    print(f"\nDistribution by dependency count:")
    for d in sorted(dep_counts.keys()):
        print(f"  {d} deps: {dep_counts[d]}")
    print(f"\nOutput saved to: {output_path}")

    return total_samples


def main():
    parser = argparse.ArgumentParser(
        description="Build swe_bench_dep dataset from SWE-Gym-Raw (parallel version)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SWE-Gym/SWE-Gym-Raw",
        help="HuggingFace dataset name (default: SWE-Gym/SWE-Gym-Raw)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--sample-per-repo",
        type=int,
        default=50,
        help="Maximum number of functions to sample per repo (default: 50)",
    )
    parser.add_argument(
        "--min-deps",
        type=int,
        default=1,
        help="Minimum number of dependencies (default: 1)",
    )
    parser.add_argument(
        "--max-deps",
        type=int,
        default=5,
        help="Maximum number of dependencies (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resource/swe_bench_dep_data.jsonl",
        help="Output file path (default: resource/swe_bench_dep_data.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=0,
        help="Maximum number of repos to process, 0 means no limit (default: 0)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Maximum total instances to generate, 0 means no limit (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    # Make output path relative to script directory if not absolute
    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(script_dir, args.output)

    build_dataset_parallel(
        dataset_name=args.dataset_name,
        split=args.split,
        sample_per_repo=args.sample_per_repo,
        min_deps=args.min_deps,
        max_deps=args.max_deps,
        output_path=args.output,
        seed=args.seed,
        max_repos=args.max_repos,
        max_instances=args.max_instances,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
