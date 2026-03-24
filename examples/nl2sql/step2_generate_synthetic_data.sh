#!/bin/sh
# ===========================================================================
# Step 2: Generate Synthetic NL2SQL Training Data (Self-Questioning)
#
# Runs the Task Manager in standalone mode to:
#   1. Explore databases via the Env Service (must be running from Step 1)
#   2. Summarize exploration trajectories into NL2SQL tasks
#   3. Filter and save synthetic data to disk
#
# This step is for **iterating on data quality** before full training.
# Check the output JSON to verify generated questions + SQL are reasonable.
#
# Shell: POSIX sh（也可用 bash 执行）
#
# Prerequisites:
#   - Step 1 (Env Service) is running
#   - DASHSCOPE_API_KEY / OPENAI_API_KEY is set (for the exploration LLM)
#
# Usage:
#   export NL2SQL_PORT=8081   # if Step 1 used a non-default port
#   sh examples/nl2sql/step2_generate_synthetic_data.sh
# Or: export NL2SQL_ENV_URL=http://127.0.0.1:8081
# 默认只加载 8 条种子任务；全量请 export NL2SQL_MAX_SEED_TASKS=  或 unset NL2SQL_MAX_SEED_TASKS
# ===========================================================================
set -eu

PROJECT_DIR=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_DIR"

# ---- Configuration ----

# Env Service URL (must match Step 1). Override with NL2SQL_ENV_URL, or set NL2SQL_HOST + NL2SQL_PORT.
NL2SQL_HOST="${NL2SQL_HOST:-127.0.0.1}"
NL2SQL_PORT="${NL2SQL_PORT:-8080}"
ENV_URL="${NL2SQL_ENV_URL:-http://${NL2SQL_HOST}:${NL2SQL_PORT}}"

# LLM for exploration and summarization (DashScope model name)
LLM_CLIENT="${NL2SQL_LLM_CLIENT:-qwen-plus}"

# Tokenizer only (NOT the exploration model). Step 2 loads this path via HF — **no Hub download** if local.
# If unset, Hydra uses actor_rollout_ref.model.path (often hits HuggingFace).
# Example (your machine):
#   export NL2SQL_TOKENIZER_MODEL_PATH=/data1/lhy/model_weight/Qwen3-4B-Instruct-2507
# For Qwen3 checkpoints, also set NL2SQL_USE_QWEN3=1 so rollout matches Qwen3 (/no_think, etc.).
NL2SQL_TOKENIZER_MODEL_PATH="${NL2SQL_TOKENIZER_MODEL_PATH:-}"
NL2SQL_USE_QWEN3="${NL2SQL_USE_QWEN3:-0}"

# Number of exploration rounds per seed task
N_EXPLORATIONS="${NL2SQL_N_EXPLORATIONS:-2}"

# Max steps per exploration episode
MAX_EXPLORE_STEP="${NL2SQL_MAX_EXPLORE_STEP:-20}"

# Parallel exploration threads
NUM_THREADS="${NL2SQL_NUM_THREADS:-8}"

# 种子任务条数上限（经 NL2SQL_MAX_SEED_TASKS 传给 Task Manager / Env get_env_profile）。
# 未设置时默认 8（快速试跑）；全量跑请在执行前 export NL2SQL_MAX_SEED_TASKS=  设为空，或 unset NL2SQL_MAX_SEED_TASKS。
# 其它试跑规模：export NL2SQL_MAX_SEED_TASKS=16
NL2SQL_MAX_SEED_TASKS="${NL2SQL_MAX_SEED_TASKS-8}"
export NL2SQL_MAX_SEED_TASKS

# Output paths for generated data
TRAIN_OUTPUT="${NL2SQL_TRAIN_OUTPUT:-nl2sql_explored.train.json}"
VAL_OUTPUT="${NL2SQL_VAL_OUTPUT:-nl2sql_explored.val.json}"

CONFIG_PATH="$PROJECT_DIR/config"

echo "============================================"
echo " NL2SQL Self-Questioning: Synthetic Data Gen"
echo "============================================"
echo " Env Service   : $ENV_URL"
echo " LLM           : $LLM_CLIENT"
echo " Explorations  : $N_EXPLORATIONS per seed"
echo " Max steps     : $MAX_EXPLORE_STEP"
echo " Threads       : $NUM_THREADS"
if [ -n "$NL2SQL_MAX_SEED_TASKS" ]; then
  echo " Max seed tasks: $NL2SQL_MAX_SEED_TASKS (smaller = faster debug run)"
else
  echo " Max seed tasks: <all — can be very slow at 0% tqdm>"
fi
echo " Output (train): $TRAIN_OUTPUT"
echo " Output (val)  : $VAL_OUTPUT"
if [ -n "$NL2SQL_TOKENIZER_MODEL_PATH" ]; then
  echo " Tokenizer HF  : $NL2SQL_TOKENIZER_MODEL_PATH (local; overrides actor model for Step 2)"
else
  echo " Tokenizer HF  : <from actor_rollout_ref.model.path — may download from HuggingFace>"
fi
echo " Qwen3 rollout : NL2SQL_USE_QWEN3=$NL2SQL_USE_QWEN3 (set to 1 for Qwen3-* tokenizer dirs)"
echo "============================================"
echo ""

# Build argv without bash arrays (POSIX)
set -- python3 -m agentevolver.module.task_manager \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_type=nl2sql \
    env_service.env_url="$ENV_URL" \
    task_manager.llm_client="$LLM_CLIENT" \
    task_manager.n="$N_EXPLORATIONS" \
    task_manager.env_profile=cookbook/env_profiles/nl2sql.json \
    task_manager.train_data_path="$TRAIN_OUTPUT" \
    task_manager.val_data_path="$VAL_OUTPUT" \
    task_manager.num_explore_threads="$NUM_THREADS" \
    task_manager.strategy=random \
    task_manager.strategy_args.max_explore_step="$MAX_EXPLORE_STEP" \
    task_manager.strategy_args.max_llm_retries=6 \
    task_manager.strategy_args.env_url="$ENV_URL" \
    task_manager.strategy_args.exploration_llm_temperature=0.8 \
    task_manager.strategy_args.exploration_llm_top_p=0.95 \
    task_manager.strategy_args.exploration_llm_top_k=50 \
    task_manager.grader.original_grader=env \
    task_manager.grader.synthetic_grader=llm \
    task_manager.mixture.use_original_tasks=true \
    task_manager.mixture.synthetic_data_ratio=0.3 \
    task_manager.mixture.shuffle=true

if [ -n "$NL2SQL_TOKENIZER_MODEL_PATH" ]; then
  set -- "$@" task_manager.tokenizer_model_path="$NL2SQL_TOKENIZER_MODEL_PATH"
fi
if [ "$NL2SQL_USE_QWEN3" = "1" ]; then
  set -- "$@" actor_rollout_ref.rollout.use_qwen3=true
fi

# Capture python exit status under POSIX sh (pipeline exit would be tee's otherwise)
_run_log="log_nl2sql_datagen_$(date +%Y%m%d_%H%M%S).log"
_tmp="${TMPDIR:-/tmp}/nl2sql_step2_$$.tmp"
: >"$_tmp"
trap 'rm -f "$_tmp"' EXIT INT HUP
set +e
"$@" >"$_tmp" 2>&1
_py_rc=$?
set -e
tee "$_run_log" <"$_tmp"

echo ""
echo "============================================"
echo " Data generation complete."
echo " Train data: $TRAIN_OUTPUT"
echo " Val data:   $VAL_OUTPUT"
echo ""
echo " Inspect the generated JSON to verify quality."
echo " When satisfied, proceed to Step 3 for training."
echo "============================================"

exit "$_py_rc"
