#!/bin/sh
# Launch the NL2SQL environment service for AgentEvolver.
#
# Shell: POSIX sh（也可用 bash 执行）
#
# Required environment variables (set before running):
#   NL2SQL_DATA_PATH  – path to BIRD/Spider-style JSON
#   NL2SQL_DB_DIR     – directory containing database files
#   NL2SQL_DB_TYPE    – "duckdb" (default) or "sqlite"
# Optional:
#   NL2SQL_HOST       – bind address (default 127.0.0.1)
#   NL2SQL_PORT       – HTTP port (default 8080)
#
# Usage:
#   source $(conda info --base)/etc/profile.d/conda.sh
#   conda activate agentevolver_nl2sql   # or your env name
#   cd env_service/launch_script
#   sh nl2sql.sh
set -eu

# Default values — override with env vars
export NL2SQL_DATA_PATH="${NL2SQL_DATA_PATH:?'Set NL2SQL_DATA_PATH to your training JSON'}"
export NL2SQL_DB_DIR="${NL2SQL_DB_DIR:?'Set NL2SQL_DB_DIR to your database directory'}"
export NL2SQL_DB_TYPE="${NL2SQL_DB_TYPE:-duckdb}"

HOST="${NL2SQL_HOST:-127.0.0.1}"
PORT="${NL2SQL_PORT:-8080}"

exec python -m env_service.env_service --env nl2sql --portal "$HOST" --port "$PORT"
