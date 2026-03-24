# NL2SQL Self-Questioning 示例

本文档介绍如何使用 AgentEvolver 的 **Self-Questioning** 机制为 NL2SQL（自然语言转 SQL）任务自动生成训练数据，并用于 GRPO 强化学习训练。

---

## 背景

NL2SQL 训练需要大量高质量的 **(自然语言问题, SQL 查询)** 配对数据。传统做法依赖人工标注，成本高且难以覆盖多样的查询模式。

Self-Questioning 提供了一条自动化路径：**让 Agent 自主探索数据库，从探索轨迹中提炼出新的 NL2SQL 训练样本**。整个过程无需人工标注。

---

## 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Questioning for NL2SQL                   │
│                                                                 │
│  ┌──────────┐     ┌───────────┐     ┌───────────┐              │
│  │ 种子任务  │────▶│  数据库   │────▶│  探索轨迹  │              │
│  │ (BIRD/   │     │  探索环境  │     │ (schema + │              │
│  │  Spider) │     │           │     │  SQL 尝试) │              │
│  └──────────┘     └───────────┘     └─────┬─────┘              │
│                                           │                     │
│                                     LLM 总结提炼                │
│                                           │                     │
│                                    ┌──────▼──────┐              │
│                                    │  合成 NL2SQL │              │
│                                    │  (问题 + SQL)│              │
│                                    └──────┬──────┘              │
│                                           │                     │
│                          过滤（去重 + 可行性验证）               │
│                                           │                     │
│                        ┌──────────────────▼──────────────────┐  │
│                        │ 混合数据集 (原始 70% + 合成 30%)     │  │
│                        └──────────────────┬──────────────────┘  │
│                                           │                     │
│                                    GRPO 训练                    │
└─────────────────────────────────────────────────────────────────┘
```

**关键步骤说明：**

1. **数据库探索** — 探索 Agent 连接到一个真实数据库，使用 4 个工具（查看 schema、采样数据、验证 SQL、执行 SQL）来深入理解数据库结构和内容。

2. **任务提炼** — 一个 LLM 总结器分析探索轨迹，从中抽象出真实用户可能提出的自然语言问题，以及对应的 SQL 查询作为参考答案。

3. **质量过滤** — 自动去除重复任务和不可执行的 SQL，保留高质量样本。

4. **数据混合** — 合成数据与原始 BIRD/Spider 标注数据按比例混合，用于 GRPO 训练。

---

## 项目结构

本示例涉及的文件：

```
AgentEvolver/
├── env_service/environments/nl2sql/
│   ├── __init__.py
│   ├── nl2sql_env.py              # NL2SQL 环境适配层（核心）
│   └── setup.sh                   # 依赖安装
├── env_service/launch_script/
│   └── nl2sql.sh                  # 环境服务启动脚本
├── cookbook/env_profiles/
│   └── nl2sql.json                # NL2SQL 领域描述 + 任务生成规则
├── examples/nl2sql/
│   ├── step1_start_env_service.sh # 步骤 1：启动环境服务
│   ├── step2_generate_synthetic_data.sh  # 步骤 2：生成合成数据
│   └── step3_train_grpo.sh        # 步骤 3：GRPO 训练
└── examples/nl2sql.yaml           # Hydra 训练配置
```

---

## 前置准备

### 1. 数据集

下载 BIRD 或 Spider 数据集，你需要：

- **训练 JSON** — 包含 `question_id`、`question`、`sql`（或 `SQL`）、`db_id`、`evidence`（可选）、`dataset_type`（可选）字段的列表。
- **数据库文件** — 对应的 DuckDB (`.db`) 或 SQLite (`.sqlite`) 数据库文件。

JSON 示例（一条记录）：

```json
{
  "question_id": 42,
  "question": "How many employees earn more than the average salary?",
  "sql": "SELECT COUNT(*) FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
  "db_id": "company",
  "evidence": "salary is stored in the employees table",
  "dataset_type": "bird"
}
```

数据库文件按如下命名规则存放在同一目录下：

| 数据集 | 数据库类型 | 文件名格式 |
|--------|-----------|-----------|
| BIRD | DuckDB | `bird_{db_id}.db` |
| BIRD | SQLite | `bird_{db_id}.sqlite` |
| Spider | SQLite | 标准发布包：`database/{db_id}/{db_id}.sqlite`（`NL2SQL_DB_DIR` 指向 Spider 根目录，如 `spider-data/`）。也支持旧布局：`Spider_sqlite/spider_sqlite/spider_{db_id}.sqlite`；默认 **`NL2SQL_SPIDER_PATH_STYLE=auto`** 会在两者间自动选择（优先标准布局）。 |

若使用官方 `dev.json` / `train_spider.json`（字段为 `query` 且无 `question_id`），请设置：

```bash
export NL2SQL_DB_TYPE=sqlite
export NL2SQL_DEFAULT_DATASET_TYPE=spider   # 行内无 dataset_type 时按 Spider 解析路径
```

每行会自动分配 `task_id`：`row_0`, `row_1`, …；SQL 取自 `query` 字段（`sql` 为解析树对象时忽略）。

### 2. 安装依赖

```bash
cd env_service/environments/nl2sql
bash setup.sh
```

或手动安装：

```bash
pip install duckdb sqlglot fastapi uvicorn ray loguru pydantic requests
```

### 3. API Key

Self-Questioning 的探索和总结阶段需要调用 LLM。设置 API Key：

```bash
export DASHSCOPE_API_KEY="your-api-key"
# 或
export OPENAI_API_KEY="your-api-key"
```

---

## 服务端口（可自定义）

Env Service 的监听地址由环境变量控制（`step1_start_env_service.sh` 与 `env_service/launch_script/nl2sql.sh` 均支持）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NL2SQL_HOST` | `127.0.0.1` | 绑定 IP |
| `NL2SQL_PORT` | `8080` | HTTP 端口；**若 8080 已被占用**（如 `address already in use`），改为 `8081` 等 |
| `NL2SQL_ENV_URL` | 无 | **Step 2 / Step 3** 连接服务用的完整 URL。不设置时，脚本使用 `http://${NL2SQL_HOST}:${NL2SQL_PORT}` |

**建议**：在步骤 1～3 的终端里使用**同一套** `NL2SQL_HOST` / `NL2SQL_PORT`（或统一设置 `NL2SQL_ENV_URL`），避免连错端口。

使用 Hydra 跑训练时，`examples/nl2sql.yaml` 里的 `env_service.env_url` 必须与 Step 1 实际地址一致，例如端口改为 8081：

```bash
python main_xxx.py env_service.env_url=http://127.0.0.1:8081
```

（将 `main_xxx.py` 换成你仓库中实际的训练入口脚本名。）

---

## 快速上手（三步完成）

`examples/nl2sql/step1/2/3` 与 `env_service/launch_script/nl2sql.sh` 均为 **POSIX sh** 脚本，**`sh …` 与 `bash …` 均可**。

### 步骤 1：启动 NL2SQL 环境服务

在 **Terminal 1** 中：

```bash
# 设置数据路径
export NL2SQL_DATA_PATH=/path/to/bird_dev.json
export NL2SQL_DB_DIR=/path/to/databases
export NL2SQL_DB_TYPE=duckdb   # 或 sqlite

# 可选：8080 被占用时改用其它端口（Step 2/3 需使用同一端口或设置 NL2SQL_ENV_URL）
export NL2SQL_PORT=8081
# export NL2SQL_HOST=127.0.0.1   # 一般不用改

# 启动服务
sh examples/nl2sql/step1_start_env_service.sh
```

服务将在 `http://${NL2SQL_HOST:-127.0.0.1}:${NL2SQL_PORT:-8080}` 启动。**保持此终端运行。**

健康检查（与 `NL2SQL_PORT` 一致，**未设置时默认为 8080**）：

```bash
curl "http://127.0.0.1:${NL2SQL_PORT:-8080}/healthz"
```

例如本机改用 8081 端口时：

```bash
export NL2SQL_PORT=8081
curl http://127.0.0.1:8081/healthz
```

### 步骤 2：生成合成数据

在 **Terminal 2** 中（**与 Step 1 相同的端口**，例如也 `export NL2SQL_PORT=8081`）：

```bash
export NL2SQL_PORT=8081   # 与 Step 1 一致；默认 8080 时可省略
sh examples/nl2sql/step2_generate_synthetic_data.sh
```

**只做 Step 2、暂不训练时**：探索与总结仍走 **`DASHSCOPE_API_KEY` + `task_manager.llm_client`（如 qwen-plus）**，但 Task Manager 还会用 **Hugging Face 上的 tokenizer** 做长度相关逻辑，默认路径来自配置里的 **`actor_rollout_ref.model.path`**（常为 `Qwen/Qwen2.5-14B-Instruct`），**无外网时会报错**。请在本机准备一份**已下载好的、尽量小的同族模型目录**（仅用于 tokenizer，例如 Qwen2.5-0.5B-Instruct 快照），并设置：

```bash
export NL2SQL_TOKENIZER_MODEL_PATH=/path/to/local/Qwen2.5-0.5B-Instruct
sh examples/nl2sql/step2_generate_synthetic_data.sh
```

或在 Hydra 中覆盖：`task_manager.tokenizer_model_path=/path/to/local/...`。

此脚本会：
1. 遍历训练集中的种子任务
2. 对每个任务对应的数据库进行探索
3. 从探索轨迹中提炼出新的 NL2SQL 问题
4. 过滤低质量数据并保存

完成后检查输出文件（默认 `nl2sql_explored.train.json`），确认生成的问题和 SQL 是否合理。

**建议**：第一次运行时先设少量数据观察质量，再调整参数大批量生成。可调节的关键参数：

```bash
export NL2SQL_N_EXPLORATIONS=2       # 每个种子探索几轮
export NL2SQL_MAX_EXPLORE_STEP=20    # 每轮探索最多几步
export NL2SQL_NUM_THREADS=8          # 并行线程数
```

### 步骤 3：启动 GRPO 训练

合成数据质量满意后，启动训练：

```bash
export NL2SQL_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

sh examples/nl2sql/step3_train_grpo.sh
```

训练脚本会：
- 加载步骤 2 生成的合成数据
- 与原始 BIRD/Spider 数据按 7:3 比例混合
- 使用 GRPO 算法进行强化学习训练

---

## 探索工具说明

NL2SQL 环境向探索 Agent 提供了 4 个工具，Agent 通过 Python 函数调用语法使用它们：

### `get_table_metadata(table_name)`

返回指定表的结构信息（列名、类型、主键、非空约束）。

```python
get_table_metadata("employees")
```

输出示例：

```
**Table: employees** (5 columns)

| Column | Type | PK | Not Null |
| --- | --- | --- | --- |
| id | INTEGER | Yes | Yes |
| name | TEXT | | Yes |
| department | TEXT | | |
| salary | REAL | | |
| hire_date | DATE | | |
```

### `get_sample_rows(table_name, n=5)`

获取表的样例数据，帮助理解真实数据内容和格式。

```python
get_sample_rows("employees", 3)
```

### `is_sql_executable(sql)`

使用 EXPLAIN 验证 SQL 是否语法正确且可执行。

```python
is_sql_executable("SELECT name FROM employees WHERE salary > 50000")
```

### `test_sql(sql, n=5)`

实际执行 SQL 并返回结果（仅允许 SELECT/WITH 查询，自动限制返回行数）。

```python
test_sql("SELECT department, AVG(salary) as avg_sal FROM employees GROUP BY department ORDER BY avg_sal DESC", 10)
```

---

## Environment Profile 说明

`cookbook/env_profiles/nl2sql.json` 描述了 NL2SQL 领域的语义结构，分为三部分：

### 实体定义

- **Database Schema** — 数据库结构（表、列、键），对应工具 `get_table_metadata` 和 `get_sample_rows`
- **SQL Query** — SQL 查询语句，对应工具 `is_sql_executable` 和 `test_sql`

### 任务偏好

```json
{
  "num_entities": 2,          // 任务涉及 schema理解 + SQL编写
  "num_opts": 3,              // 平均涉及 3 个操作步骤
  "relation_difficulty": 3    // Hard：需要跨表推理
}
```

### Rubrics（任务生成规则）

Profile 中定义了 7 条 Rubrics，指导 LLM 总结器生成高质量的 NL2SQL 任务：

1. 每个任务必须是可用单条 SELECT 查询回答的自然语言问题
2. `action_sequence` 必须包含可执行的 SQL
3. 问题应像真实的分析/业务问题，而非技术性 SQL 描述
4. 包含具体值（名称、日期、阈值）使问题可验证
5. 倾向生成需要 JOIN、GROUP BY、HAVING、子查询的任务
6. 不生成修改数据的任务或简单的 `SELECT *` 任务
7. 覆盖不同的表、查询模式和难度级别

你可以编辑 `nl2sql.json` 中的 `rubrics` 数组来调整生成策略。

---

## 关键配置参数

| 参数 | 位置 | 推荐值 | 说明 |
|------|------|--------|------|
| `task_manager.n` | YAML / 脚本 | 2 | 每个种子任务探索几轮；越大越多样，成本越高 |
| `strategy_args.max_explore_step` | YAML / 脚本 | 20 | 每轮探索最大步数；数据库场景通常 15-25 步即可 |
| `strategy_args.exploration_llm_temperature` | YAML | 0.8 | 探索多样性；偏高有利于发现更多查询模式 |
| `mixture.synthetic_data_ratio` | YAML / 脚本 | 0.3 | 合成数据占比；建议从小比例开始，逐步增加 |
| `grader.synthetic_grader` | YAML | `llm` | 合成任务的评判方式；LLM 对比参考 SQL 打分 |
| `NL2SQL_DB_TYPE` | 环境变量 | `duckdb` | 数据库引擎类型 |
| `NL2SQL_PORT` / `NL2SQL_HOST` | 环境变量 | `8080` / `127.0.0.1` | Env Service 绑定；Step 2/3 默认按二者拼 `env_url` |
| `NL2SQL_ENV_URL` | 环境变量 | 由 HOST+PORT 推导 | 显式指定 Env Service 根 URL（覆盖 HOST+PORT） |

---

## 常见问题

### Q: 如何验证环境服务是否正常工作？

启动服务后，使用 curl 手动测试交互流程（将 `8081` 换成你的 `NL2SQL_PORT`，默认 `8080`）：

```bash
BASE=http://127.0.0.1:8081

curl -sS "${BASE}/healthz"

# 创建实例
curl -X POST "${BASE}/create" \
  -H "Content-Type: application/json" \
  -d '{"env_type":"nl2sql","task_id":"1","instance_id":"test-001"}'

# 执行一步
curl -X POST "${BASE}/step" \
  -H "Content-Type: application/json" \
  -d '{"instance_id":"test-001","messages":{"role":"assistant","content":"```python\nget_table_metadata(\"employees\")\n```"}}'

# 释放实例
curl -X POST "${BASE}/release" \
  -H "Content-Type: application/json" \
  -d '{"instance_id":"test-001"}'
```

### Q: 生成的合成数据质量不理想怎么办？

1. **调整 Rubrics** — 编辑 `cookbook/env_profiles/nl2sql.json` 中的 `rubrics` 数组，添加更具体的约束
2. **增加探索步数** — 提高 `max_explore_step`，让 Agent 更深入地探索数据库
3. **提高 LLM 能力** — 将 `llm_client` 换成更强的模型（如 `qwen-max`）
4. **调节温度** — 降低 `exploration_llm_temperature` 可使探索更集中，升高则更发散

### Q: 能否只用合成数据训练，不混合原始数据？

可以，调整配置：

```yaml
task_manager:
  mixture:
    use_original_tasks: false
    synthetic_data_ratio: 999999.0  # 实际效果为 100% 合成数据
```

### Q: 支持哪些数据库？

当前支持 **DuckDB** 和 **SQLite**（只读模式）。通过 `NL2SQL_DB_TYPE` 环境变量切换。如需支持其他数据库，可扩展 `nl2sql_env.py` 中的 `_connect` 函数。

---

## 进阶使用

### 使用 launcher.py 一键启动

如果你配置了 `.env` 文件，可以通过 launcher 自动管理服务生命周期：

```bash
python launcher.py --conf examples/nl2sql.yaml --with-nl2sql
```

需要在 `.env` 中添加对应配置（参考 `example.env`）。

### 结合 Self-Navigating / Self-Attributing

在 `examples/nl2sql.yaml` 中开启 `exp_manager` 和 `attribution_driven_credit_assignment` 即可叠加 AgentEvolver 的其他两个自进化机制。详见 [Experience Manager](exp_manager.md) 和 [Advantage Processor](adv_processor.md) 文档。
