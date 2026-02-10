# Hybrid Gym: Dependency Search

Evaluates an agent's ability to **find all functions/classes directly called by a target function** and add a comment above each dependency's definition. The agent must analyze call relationships, resolve definitions, and annotate without modifying code.

Dataset: [hybrid-gym/hybrid_gym_dep_search](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_dep_search) (4,000 instances)

## Setup

Please follow instructions [here](../../README.md#setup) to setup your local development environment and LLM.

## Run Inference

```bash
poetry run python evaluation/benchmarks/hybrid_gym_dep_search/run_infer.py \
    --llm-config o3-mini \
    --agent-cls CodeActAgent \
    --max-iterations 30 \
    --eval-num-workers 3 \
    --eval-n-limit 10 \
    --dataset hybrid-gym/hybrid_gym_dep_search \
    --split train
```

## Evaluate

```bash
poetry run python evaluation/benchmarks/hybrid_gym_dep_search/eval_dep.py \
    --output-file <path/to/output.jsonl> \
    --data-file hybrid-gym/hybrid_gym_dep_search
```

**Metrics:** Precision, recall, F1 over dependencies. Full success requires: all dependencies found, no false positives, no duplicates, and comments only (no code changes).

## Build Dataset (Preprocessing)

The preprocessing scripts build the ground-truth dataset by analyzing repositories from SWE-Gym using [Jedi](https://jedi.readthedocs.io/) for scope-aware dependency resolution.

### Requirements

```bash
pip install jedi datasets tqdm
```

### Standard Version

```bash
cd evaluation/benchmarks/hybrid_gym_dep_search

python3 preprocess/build_dataset.py \
    --max-instances 100 \
    --min-deps 1 \
    --max-deps 3 \
    --sample-per-repo 10 \
    --output resource/swe_bench_dep_data.jsonl
```

### Parallel Version (Faster)

```bash
python3 preprocess/build_dataset_parallel.py \
    --max-instances 4000 \
    --min-deps 1 \
    --max-deps 3 \
    --sample-per-repo 20 \
    --output resource/swe_bench_dep_data_full.jsonl \
    --num-workers 8
```

### Build Dataset Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-name` | `SWE-Gym/SWE-Gym-Raw` | HuggingFace dataset name |
| `--split` | `train` | Dataset split to process |
| `--sample-per-repo` | `50` | Max functions to sample per repo |
| `--min-deps` | `1` | Minimum dependencies required |
| `--max-deps` | `5` | Maximum dependencies allowed |
| `--max-repos` | `0` | Max repos to process (0 = no limit) |
| `--max-instances` | `0` | Max total instances (0 = no limit) |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-workers` | `8` | Parallel workers (parallel version only) |

**Balanced Sampling**: When `--max-instances` is set, the script uses round-robin sampling that prioritizes underrepresented dependency counts, ensuring a balanced distribution.
