# Hybrid Gym: Function Generation

Evaluates an agent's ability to **implement a function body** given its signature and docstring. The function body is masked with `pass  # TODO`, and the agent must write a correct implementation.

Dataset: [hybrid-gym/hybrid_gym_func_gen](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_gen) (7,415 instances)

## Setup

Please follow instructions [here](../../README.md#setup) to setup your local development environment and LLM.

**Evaluation requires the `yiqingxyq/repost:v0` Docker image.** Pull it from [Docker Hub](https://hub.docker.com/r/yiqingxyq/repost):

```bash
docker pull yiqingxyq/repost:v0
```

## Run Inference

```bash
./evaluation/benchmarks/hybrid_gym_func_gen/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [split]

# Example
./evaluation/benchmarks/hybrid_gym_func_gen/scripts/run_infer.sh llm.o3-mini HEAD CodeActAgent 10 30 3
```

Or run directly:

```bash
poetry run python evaluation/benchmarks/hybrid_gym_func_gen/run_infer_no_image.py \
    --llm-config o3-mini \
    --agent-cls CodeActAgent \
    --max-iterations 30 \
    --eval-num-workers 3 \
    --eval-n-limit 10 \
    --dataset hybrid-gym/hybrid_gym_func_gen \
    --split train
```

## Evaluate

```bash
poetry run python evaluation/benchmarks/hybrid_gym_func_gen/eval_func_completion.py \
    --output-file <path/to/output.jsonl> \
    --use-docker
```

**Metric:** Test pass rate (generated function produces correct output on all test cases).
