#!/usr/bin/env bash
# Setup script for the NL2SQL environment in AgentEvolver.
#
# Prerequisites:
#   - conda is installed and available
#   - The BIRD / Spider dataset JSON and database files are downloaded
#
# Usage:
#   cd env_service/environments/nl2sql
#   bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- dependencies ----
pip install duckdb sqlglot fastapi uvicorn ray loguru pydantic requests

echo ""
echo "============================================"
echo " NL2SQL environment setup complete."
echo ""
echo " Before launching, set the following env vars:"
echo "   export NL2SQL_DATA_PATH=/path/to/bird_dev.json"
echo "   export NL2SQL_DB_DIR=/path/to/databases"
echo "   export NL2SQL_DB_TYPE=duckdb          # or sqlite"
echo "============================================"
