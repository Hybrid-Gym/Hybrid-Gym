# Hybrid Gym: Function Localization

Evaluates an agent's ability to **locate a function by its description** (no file path or name given) and **add a docstring** to it. The agent must search the codebase, find the target function, and insert an appropriate docstring without modifying any code.

Dataset: [hybrid-gym/hybrid_gym_func_localize](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_localize) (5,000 instances)

## Setup

Please follow instructions [here](../../README.md#setup) to setup your local development environment and LLM.

## Run Inference

```bash
./evaluation/benchmarks/hybrid_gym_func_localize/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [split]

# Example
./evaluation/benchmarks/hybrid_gym_func_localize/scripts/run_infer.sh llm.o3-mini HEAD CodeActAgent 10 30 3
```

Or run directly:

```bash
poetry run python evaluation/benchmarks/hybrid_gym_func_localize/run_infer_no_image.py \
    --llm-config o3-mini \
    --agent-cls CodeActAgent \
    --max-iterations 30 \
    --eval-num-workers 3 \
    --eval-n-limit 10 \
    --dataset hybrid-gym/hybrid_gym_func_localize \
    --split train
```

## Evaluate

```bash
poetry run python evaluation/benchmarks/hybrid_gym_func_localize/eval_localize.py \
    --output-file <path/to/output.jsonl> \
    --data-file hybrid-gym/hybrid_gym_func_localize
```

**Metric:** Success requires both: (1) target function's docstring was edited, and (2) all changes are comments/docstrings only (no code modifications).

## Build Dataset (Preprocessing)

The preprocessing scripts build the dataset by extracting functions with docstrings from SWE-Gym repositories and using an LLM to generate functionality descriptions.

### Step 1: Generate filtered.jsonl (from SWE-Gym-Raw)

Extracts all functions with docstrings from repos in SWE-Gym-Raw and uses an LLM to generate a functionality description for each.

```bash
export OPENAI_API_KEY=<your-key>

poetry run python preprocess/get_docstring.py \
    --dataset-name SWE-Gym/SWE-Gym-Raw \
    --split train \
    --mode all_func \
    --openai-model gpt-4o-mini \
    --sample-size 50  # number of repos to process (0 = all)
```

### Step 2: Generate masked data (optional, for harder evaluation)

Generate descriptions that do NOT mention the function/class name:

```bash
# Convert existing data to masked format
poetry run python preprocess/convert_to_masked.py \
    --input-file resource/subsets/swe_doc_gen_locate_20.jsonl \
    --output-file resource/swe_doc_gen_masked_20.jsonl \
    --model gpt-4o-mini

# Or generate from scratch
poetry run python preprocess/get_docstring_masked.py \
    --dataset-name SWE-Gym/SWE-Gym-Raw \
    --split train \
    --sample-size 20 \
    --output-file resource/swe_doc_gen_masked_20.jsonl
```

### Step 3: Generate subsets of any size

```bash
poetry run python preprocess/generate_subset.py \
    --input-file resource/swe_doc_gen_filtered.jsonl \
    --n-repos 50 \
    --output-dir resource/subsets \
    --seed 42
```

### Step 4: Sample instances per repo

```bash
poetry run python preprocess/sample_instances.py \
    --input-file resource/swe_doc_gen_filtered.jsonl \
    --sample-size 10
```
