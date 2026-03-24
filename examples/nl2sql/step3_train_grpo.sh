#!/bin/sh
# ===========================================================================
# Step 3: GRPO Training with Self-Questioning Synthetic Data
#
# Runs the full AgentEvolver GRPO training pipeline for NL2SQL,
# mixing original BIRD/Spider tasks with synthetic data from Step 2.
#
# Shell: POSIX sh（也可用 bash 执行）
#
# Prerequisites:
#   - Step 1 (Env Service) is running
#   - Step 2 has generated synthetic data (or set N > 0 to generate on-the-fly)
#   - Model weights are available locally
#
# Usage:
#   export NL2SQL_PORT=8081   # if Step 1 used a non-default port
#   sh examples/nl2sql/step3_train_grpo.sh
# Or: export NL2SQL_ENV_URL=http://127.0.0.1:8081
# ===========================================================================
set -eu

PROJECT_DIR=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_DIR"

# ---- Configuration (edit these) ----

# Model path (local HuggingFace model directory)
MODEL_PATH="${NL2SQL_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Env Service URL (must match Step 1). Override with NL2SQL_ENV_URL, or set NL2SQL_HOST + NL2SQL_PORT.
NL2SQL_HOST="${NL2SQL_HOST:-127.0.0.1}"
NL2SQL_PORT="${NL2SQL_PORT:-8080}"
ENV_URL="${NL2SQL_ENV_URL:-http://${NL2SQL_HOST}:${NL2SQL_PORT}}"

# Pre-generated synthetic data from Step 2 (set to null for on-the-fly generation)
TRAIN_DATA="${NL2SQL_TRAIN_DATA:-nl2sql_explored.train.json}"
VAL_DATA="${NL2SQL_VAL_DATA:-nl2sql_explored.val.json}"

# Number of GPUs
N_GPUS="${NL2SQL_N_GPUS:-1}"

# Experiment naming
PROJECT_NAME="${NL2SQL_PROJECT_NAME:-nl2sql_self_questioning}"
EXPERIMENT_NAME="${NL2SQL_EXPERIMENT_NAME:-nl2sql_sq_grpo}"

CONFIG_PATH="$PROJECT_DIR/config"
current_time=$(date +%Y%m%d_%H%M%S)
log_file="log_nl2sql_train_${current_time}.log"

echo "============================================"
echo " NL2SQL GRPO Training"
echo "============================================"
echo " Model      : $MODEL_PATH"
echo " Env Service: $ENV_URL"
echo " Train data : $TRAIN_DATA"
echo " Val data   : $VAL_DATA"
echo " GPUs       : $N_GPUS"
echo " Experiment : $EXPERIMENT_NAME"
echo " Log        : $log_file"
echo "============================================"
echo ""

_tmp="${TMPDIR:-/tmp}/nl2sql_step3_$$.tmp"
: >"$_tmp"
trap 'rm -f "$_tmp"' EXIT INT HUP

set +e
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    \
    env_service.env_type=nl2sql \
    env_service.env_url="$ENV_URL" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    data.return_raw_chat=true \
    data.train_files=null \
    data.val_files=null \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.max_env_worker=16 \
    actor_rollout_ref.rollout.context_template="linear" \
    actor_rollout_ref.rollout.max_env_len=2048 \
    actor_rollout_ref.rollout.sparse=true \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.logger="['console']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.val_before_train=true \
    trainer.validation_data_dir="experiments/${EXPERIMENT_NAME}/validation_log" \
    trainer.rollout_data_dir="experiments/${EXPERIMENT_NAME}/rollout_log" \
    \
    task_manager.train_data_path="$TRAIN_DATA" \
    task_manager.val_data_path="$VAL_DATA" \
    task_manager.llm_client=qwen-plus \
    task_manager.n=0 \
    task_manager.env_profile=cookbook/env_profiles/nl2sql.json \
    task_manager.bs=16 \
    task_manager.num_explore_threads=8 \
    task_manager.mixture.use_original_tasks=true \
    task_manager.mixture.synthetic_data_ratio=0.3 \
    task_manager.mixture.shuffle=true \
    task_manager.grader.original_grader=env \
    task_manager.grader.synthetic_grader=llm \
    task_manager.strategy=random \
    task_manager.strategy_args.max_explore_step=20 \
    task_manager.strategy_args.max_llm_retries=6 \
    task_manager.strategy_args.env_url="$ENV_URL" \
    \
    exp_manager.val_rollout_mode="woexp" \
    exp_manager.train_rollout_mode="woexp" \
    exp_manager.rollout_ratio=0.0 \
    exp_manager.train_sample_mode="alldiscard" \
    exp_manager.init_exp_before_training=false \
    exp_manager.reme.enable_summarizer=false \
    exp_manager.reme.enable_context_generator=false \
    \
    attribution_driven_credit_assignment.enable=false \
    \
    >"$_tmp" 2>&1
_py_rc=$?
set -e
tee "$log_file" <"$_tmp"

echo ""
echo "============================================"
echo " Training complete. Log: $log_file"
echo "============================================"

exit "$_py_rc"
