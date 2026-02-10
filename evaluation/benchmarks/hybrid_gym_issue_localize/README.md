# Hybrid Gym: Issue Localization

Evaluates an agent's ability to **locate code that needs modification** given a GitHub issue description. The agent must search the codebase, identify relevant files and locations, and add comments explaining why each location is related to the issue. No code modifications allowed, only comments.

Dataset: [SWE-Gym/SWE-Gym-Raw](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Raw)

## Setup

Please follow instructions [here](../../README.md#setup) to setup your local development environment and LLM.

## Run Inference

```bash
./evaluation/benchmarks/hybrid_gym_issue_localize/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [split]

# Example
./evaluation/benchmarks/hybrid_gym_issue_localize/scripts/run_infer.sh llm.o3-mini HEAD CodeActAgent 10 30 3 SWE-Gym/SWE-Gym-Raw train
```

Or run directly:

```bash
poetry run python evaluation/benchmarks/hybrid_gym_issue_localize/run_infer.py \
    --llm-config o3-mini \
    --agent-cls CodeActAgent \
    --max-iterations 30 \
    --eval-num-workers 3 \
    --eval-n-limit 10 \
    --dataset SWE-Gym/SWE-Gym-Raw \
    --split train
```

## Evaluate

```bash
poetry run python evaluation/benchmarks/hybrid_gym_issue_localize/eval_localize.py \
    --output-file <path/to/output.jsonl>
```

**Metric:** Success requires both: (1) agent's patch touches at least one gold file (file-level localization hit), and (2) all changes are comments only (no code modifications).
