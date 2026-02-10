"""Generate a subset of the swe_doc_gen data for testing/evaluation.

This script creates subsets from the filtered JSONL data for both:
- swe_doc_gen_locate: Single function per instance
- swe_doc_gen_add: 2-3 functions per instance (from same repo/image_instance_id)

Usage:
    # Generate 50 samples for both tasks from the same 50 repos
    python generate_subset.py --n-repos 50 --output-dir ../resource/subsets

    # Generate from specific filtered data file
    python generate_subset.py --input-file ../resource/swe_doc_gen_filtered.jsonl --n-repos 50
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_data(input_file: str) -> tuple[list[dict], dict]:
    """Load data and group by repo and image_instance_id."""
    all_data = []
    data_by_repo = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            all_data.append(item)
            data_by_repo[item['repo']].append(item)

    return all_data, data_by_repo


def find_repos_with_multi_functions(data_by_repo: dict, min_funcs: int = 3) -> list[tuple]:
    """Find repos that have at least min_funcs functions in the same image_instance_id."""
    suitable_repos = []

    for repo, items in data_by_repo.items():
        # Group by image_instance_id
        by_image_id = defaultdict(list)
        for item in items:
            by_image_id[item['image_instance_id']].append(item)

        # Check if any image_instance_id has enough functions
        for iid, funcs in by_image_id.items():
            if len(funcs) >= min_funcs:
                suitable_repos.append((repo, iid, funcs))
                break  # Only need one valid image_instance_id per repo

    return suitable_repos


def generate_locate_sample(func_data: dict) -> dict:
    """Generate a single-function sample for the locate task."""
    return func_data.copy()


def generate_add_sample(funcs: list[dict], num_funcs: int = None) -> dict:
    """Generate a multi-function sample for the add task."""
    if num_funcs is None:
        num_funcs = random.randint(2, min(3, len(funcs)))

    selected = random.sample(funcs, num_funcs)

    # Create the combined instance
    base = selected[0]
    add_sample = {
        'instance_id': f"{base['image_instance_id']}_multi_{random.randint(0, 9999)}",
        'repo': base['repo'],
        'base_commit': base['base_commit'],
        'image_instance_id': base['image_instance_id'],
        'functions': [
            {
                'file_path': f['file_path'],
                'module_name': f['module_name'],
                'module_type': f['module_type'],
                'module_line_start': f['module_line_start'],
                'module_line_end': f['module_line_end'],
                'docstring': f['docstring'],
                'docstring_line_start': f['docstring_line_start'],
                'docstring_line_end': f['docstring_line_end'],
            }
            for f in selected
        ]
    }

    return add_sample


def main():
    parser = argparse.ArgumentParser(description='Generate subsets for swe_doc_gen benchmarks')
    parser.add_argument('--input-file', type=str,
                        default='evaluation/benchmarks/hybrid_gym_func_localize/resource/swe_doc_gen_filtered.jsonl',
                        help='Path to filtered JSONL data file')
    parser.add_argument('--n-repos', type=int, default=50,
                        help='Number of repos to sample')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: creates in resource/subsets)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--locate-only', action='store_true',
                        help='Only generate locate data')
    parser.add_argument('--add-only', action='store_true',
                        help='Only generate add data')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    print(f"Loading data from {args.input_file}...")
    all_data, data_by_repo = load_data(args.input_file)
    print(f"  Total entries: {len(all_data)}")
    print(f"  Unique repos: {len(data_by_repo)}")

    # Find repos suitable for multi-function samples
    suitable_repos = find_repos_with_multi_functions(data_by_repo, min_funcs=3)
    print(f"  Repos with >= 3 functions in same image_instance_id: {len(suitable_repos)}")

    # Sample repos
    if len(suitable_repos) < args.n_repos:
        print(f"Warning: Only {len(suitable_repos)} suitable repos available, using all")
        selected_repos = suitable_repos
    else:
        selected_repos = random.sample(suitable_repos, args.n_repos)

    print(f"\nSelected {len(selected_repos)} repos for sampling")

    # Generate samples
    locate_samples = []
    add_samples = []

    for repo, image_id, funcs in selected_repos:
        # For locate: pick one function
        locate_sample = generate_locate_sample(random.choice(funcs))
        locate_samples.append(locate_sample)

        # For add: pick 2-3 functions
        add_sample = generate_add_sample(funcs)
        add_samples.append(add_sample)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input_file).parent / 'subsets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save samples
    if not args.add_only:
        locate_file = output_dir / f'swe_doc_gen_locate_{args.n_repos}.jsonl'
        with open(locate_file, 'w') as f:
            for sample in locate_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"\nSaved {len(locate_samples)} locate samples to: {locate_file}")

    if not args.locate_only:
        add_file = output_dir / f'swe_doc_gen_add_{args.n_repos}.jsonl'
        with open(add_file, 'w') as f:
            for sample in add_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved {len(add_samples)} add samples to: {add_file}")

    # Print sample info
    print("\n=== Sample Info ===")
    if not args.add_only and locate_samples:
        print(f"Locate samples: {len(locate_samples)}")
        sample = locate_samples[0]
        print(f"  Example: {sample['instance_id']}")
        print(f"    repo: {sample['repo']}")
        print(f"    file_path: {sample['file_path']}")
        print(f"    module_name: {sample['module_name']}")

    if not args.locate_only and add_samples:
        print(f"\nAdd samples: {len(add_samples)}")
        sample = add_samples[0]
        print(f"  Example: {sample['instance_id']}")
        print(f"    repo: {sample['repo']}")
        print(f"    functions: {len(sample['functions'])}")
        for i, func in enumerate(sample['functions']):
            print(f"      {i+1}. {func['module_name']} in {func['file_path']}")


if __name__ == '__main__':
    main()
