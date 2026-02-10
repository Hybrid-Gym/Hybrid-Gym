import argparse
import json
import os
import random

HOME_DIR = os.environ.get("HOME", "")
CODE_DIR = os.path.join(HOME_DIR, "OpenHands2/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SWE-Gym/SWE-Gym-Raw",
        help="Dataset name used in the filename (must match generation script).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split used in the filename.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of instances to sample per repo (>= repo size keeps all).",
    )
    parser.add_argument(
        "--mode",
        choices=["all_func", "multi_func"],
        default="all_func",
        help="Which dataset variant to sample.",
    )
    args = parser.parse_args()

    dataset_save_name = args.dataset_name.split("/")[-1]
    save_split = f"{args.split}_sampled{args.sample_size}"
    prefix = "swe_doc_gen_locate_multi_func" if args.mode == "multi_func" else "swe_doc_gen_locate_all_func"

    existing_dataset_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/{prefix}_{dataset_save_name}_{args.split}.json",
    )

    with open(existing_dataset_path, "r", encoding="utf-8") as f:
        existing_dataset = json.load(f)

    repo_list = list({x["repo"] for x in existing_dataset})
    sampled_dataset = []
    for repo in repo_list:
        repo_instances = [x for x in existing_dataset if x["repo"] == repo]
        if len(repo_instances) <= args.sample_size:
            sampled_dataset.extend(repo_instances)
        else:
            random.seed(42)
            sampled_instances = random.sample(repo_instances, args.sample_size)
            sampled_dataset.extend(sampled_instances)

    print(f"Sampled {len(sampled_dataset)} instances from {len(repo_list)} repos")

    sampled_dataset_path = os.path.join(
        CODE_DIR,
        f"evaluation/benchmarks/hybrid_gym_func_localize/resource/{prefix}_{dataset_save_name}_{save_split}.json",
    )
    with open(sampled_dataset_path, "w", encoding="utf-8") as f:
        json.dump(sampled_dataset, f)


if __name__ == "__main__":
    main()
