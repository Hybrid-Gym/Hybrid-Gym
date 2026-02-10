export ALLHANDS_API_KEY=$REMOTE_KEY
export RUNTIME=remote
export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"

# MODEL_NAME=MODEL_SAVE_NAME="Qwen25-Coder-14B-Instruct"
MODEL_NAME="claude45_dp"
MODEL_SAVE_NAME="claude-sonnet-4-5"

# 2. Run inference (example with o3-mini on 30-instance dataset)
python evaluation/benchmarks/hybrid_gym_dep_search/run_infer.py \
    --llm-config llm.$MODEL_NAME \
    --agent-cls CodeActAgent \
    --dataset hybrid-gym/hybrid_gym_dep_search \
    --max-iterations 30 \
    --eval-num-workers 30 \
    --eval-output-dir $STORAGE_DIR/openhands/evaluation/evaluation_outputs/dependency_outputs

# 3. Evaluate results
python evaluation/benchmarks/hybrid_gym_dep_search/eval_dep.py \
    --output-file $STORAGE_DIR/openhands/evaluation/evaluation_outputs/dependency_outputs/hybrid_gym_dep_search_data_30/CodeActAgent/${MODEL_SAVE_NAME}_maxiter_30/output.jsonl \
    --data-file hybrid-gym/hybrid_gym_dep_search \
    --save-results
