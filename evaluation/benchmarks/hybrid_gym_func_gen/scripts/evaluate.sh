#!/bin/bash
# Evaluate function completion results using RepoST's eval_script
#
# Usage: ./evaluate.sh <output.jsonl> [options]
#
# Examples:
#   ./evaluate.sh experiments/my-exp/output.jsonl
#   ./evaluate.sh experiments/my-exp/output.jsonl --no-docker
#   ./evaluate.sh experiments/my-exp/output.jsonl --no-single-container
#   ./evaluate.sh experiments/my-exp/output.jsonl --timeout 120

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <output.jsonl> [options]"
    echo ""
    echo "Options:"
    echo "  --no-docker            Run tests directly without Docker (default: use Docker)"
    echo "  --no-single-container  Run each test in separate Docker container (default: single container)"
    echo "  --timeout <sec>        Timeout for each test in seconds (default: 60)"
    echo "  --backup               Create backup of output.jsonl before updating (default: no backup)"
    echo ""
    echo "By default, all tests run in a single Docker container for speed."
    echo "Use --no-single-container if you need to run multiple evaluation processes in parallel."
    echo ""
    echo "Examples:"
    echo "  $0 experiments/my-exp/output.jsonl"
    echo "  $0 experiments/my-exp/output.jsonl --no-docker"
    echo "  $0 experiments/my-exp/output.jsonl --no-single-container"
    exit 1
fi

OUTPUT_FILE=$1
shift  # Remove first argument, pass rest to Python script

# Default: use Docker, single container mode, update output.jsonl in place, no backup
python "$SCRIPT_DIR/../eval_func_completion.py" \
    --output-file "$OUTPUT_FILE" \
    --use-docker \
    --single-container \
    --update-output \
    --no-backup \
    "$@"
