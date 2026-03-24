#!/bin/sh
# ===========================================================================
# Step 1: Start the NL2SQL Environment Service
#
# This launches the Env Service that wraps databases as interactive
# environments for Self-Questioning exploration.
#
# Shell: POSIX sh（也可用 bash 执行：bash … / sh …）
#
# Prerequisites:
#   - conda env with duckdb, fastapi, uvicorn, ray, loguru installed
#     (see env_service/environments/nl2sql/setup.sh)
#   - BIRD / Spider training JSON and database files downloaded
#
# Usage:
#   cd <AgentEvolver 根目录>
#   sh examples/nl2sql/step1_start_env_service.sh
#
# 关终端后仍要跑：请自行在外层加 nohup / tmux 等，例如：
#   nohup sh examples/nl2sql/step1_start_env_service.sh >step1.log 2>&1 &
# 若需要「PID 文件 + kill」：启动时自己写入，例如：
#   nohup sh examples/nl2sql/step1_start_env_service.sh >step1.log 2>&1 & echo $! > examples/nl2sql/nl2sql_env_service.pid
#   停止：kill "$(cat examples/nl2sql/nl2sql_env_service.pid)"
# 未用 PID 文件时也可按端口（把 8081 改成你的 NL2SQL_PORT）：
#   kill "$(lsof -t -iTCP:8081 -sTCP:LISTEN)" 2>/dev/null || true
# 前台运行：在该终端 Ctrl+C 即可。
#
# 下面「默认 export」来自你本机曾用的配置；换机器请改默认值或启动前 export 覆盖。
# Bind: NL2SQL_HOST / NL2SQL_PORT（Step 2/3 的 env_url 须与端口一致）。
# ===========================================================================
set -eu

# ---- Repo root（保证 python -m env_service 可 import）----
_SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
_REPO_ROOT=$(CDPATH= cd "$_SCRIPT_DIR/../.." && pwd)
cd "$_REPO_ROOT"

# ---- Configuration（默认值 = 你之前跑通用的那套；可用 export 覆盖）----

# JiuTian Spider JSON（含 question_id / sql / db_id / dataset_type: spider）
export NL2SQL_DATA_PATH="${NL2SQL_DATA_PATH:-/data1/lhy/dataset/JiuTianDataAgent/Synthesis_nl2sql/data/0123_rl_dataset/valid_spider.json}"

# 官方 Spider 解压根目录（其下有 database/{db_id}/{db_id}.sqlite）
export NL2SQL_DB_DIR="${NL2SQL_DB_DIR:-/data1/lhy/dataset/spider-data}"

export NL2SQL_DB_TYPE="${NL2SQL_DB_TYPE:-sqlite}"

# Spider 标准目录 tree：auto 会优先用 database/{db_id}/{db_id}.sqlite
export NL2SQL_SPIDER_PATH_STYLE="${NL2SQL_SPIDER_PATH_STYLE:-auto}"

# 仅当 JSON 里没有 dataset_type 时需要（例如官方 dev.json）；当前 valid_spider 一般不用开
# export NL2SQL_DEFAULT_DATASET_TYPE="${NL2SQL_DEFAULT_DATASET_TYPE:-spider}"

# 可选：限制种子条数（无 Step2 传参时，Env 侧 get_query_list 也会读此变量）
# export NL2SQL_MAX_SEED_TASKS="${NL2SQL_MAX_SEED_TASKS:-}"

# Ray 卡在 dashboard 时可开：export ENV_SERVICE_RAY_DASHBOARD=0
# export ENV_SERVICE_RAY_DASHBOARD="${ENV_SERVICE_RAY_DASHBOARD:-}"
# 少打 HTTP 日志：export ENV_SERVICE_UVICORN_LOG=warning

# 因 8080 占用改用 8081；Step2 的 NL2SQL_PORT 须与此一致
export NL2SQL_PORT="${NL2SQL_PORT:-8081}"
export NL2SQL_HOST="${NL2SQL_HOST:-127.0.0.1}"

HOST="$NL2SQL_HOST"
PORT="$NL2SQL_PORT"

# ---- Validate ----
if [ ! -f "$NL2SQL_DATA_PATH" ]; then
    echo "ERROR: NL2SQL_DATA_PATH not found: $NL2SQL_DATA_PATH"
    echo "Please set NL2SQL_DATA_PATH to your BIRD/Spider training JSON."
    exit 1
fi
if [ ! -d "$NL2SQL_DB_DIR" ]; then
    echo "ERROR: NL2SQL_DB_DIR not found: $NL2SQL_DB_DIR"
    echo "Please set NL2SQL_DB_DIR to your database directory."
    exit 1
fi

echo "============================================"
echo " NL2SQL Env Service"
echo "============================================"
echo " DATA_PATH : $NL2SQL_DATA_PATH"
echo " DB_DIR    : $NL2SQL_DB_DIR"
echo " DB_TYPE   : $NL2SQL_DB_TYPE"
echo " SPIDER_PATH_STYLE : ${NL2SQL_SPIDER_PATH_STYLE:-auto}"
echo " DEFAULT_DATASET_TYPE : ${NL2SQL_DEFAULT_DATASET_TYPE:-<unset>}"
echo " Endpoint  : http://${HOST}:${PORT}"
echo "============================================"
echo ""

exec python -m env_service.env_service \
    --env nl2sql \
    --portal "$HOST" \
    --port "$PORT"
