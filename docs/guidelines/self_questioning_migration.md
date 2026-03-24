# Self-Questioning 迁移指南

将 AgentEvolver 的 **Self-Questioning**（Task Manager 自动生成训练任务）迁移到新场景时，按以下层次准备与修改即可。

---

## 概述

Self-Questioning 依赖一条固定链路：

```
Environment Profile → Exploration → Summarization → Filtering → Mixing → RL Dataset
```

迁移时核心是：**环境可经 Env Service 交互** + **提供与环境语义一致的 Environment Profile** + **在 YAML 中正确配置 Task Manager**。

---

## 第一层：环境接入（必须）

### 1.1 实现 `BaseEnv` 子类

在 `env_service/environments/<your_env>/` 下实现继承 `env_service.base.BaseEnv` 的类（通常文件名为 `{env_name}_env.py`，例如 `myenv_env.py`）。

需实现的方法：

| 方法 | 作用 |
|------|------|
| `__init__(task_id, instance_id)` | 初始化环境实例 |
| `get_init_state(params)` | 返回初始状态，通常含 `state`（消息列表）、`info` |
| `step(action, params)` | 执行一步，返回 `state`、`reward`、`is_terminated`、`info` |
| `evaluate(messages, params)` | 任务评分（若环境有内置评判） |
| `get_info(messages, params)` | 环境元信息 |
| `close()` | 释放资源 |
| `get_query_list(split, params)` | 返回 train/val/test 等 split 下的 **task_id 列表** |

参考实现：`env_service/environments/appworld/appworld_env.py`、`bfcl/bfcl_env.py`、`openworld/openworld_env.py`。

### 1.2 启动环境服务

在 `env_service/launch_script/` 下增加启动脚本，例如：

```bash
exec python -m env_service.env_service --env myenv --portal 127.0.0.1 --port 8080
```

服务通过 `python -m env_service.env_service` 动态加载 `env_service/environments/<env>/<env>_env.py`。更多说明见 [Env Service 指南](env_service.md)。

### 1.3 HTTP 交互约定

训练侧通过 `EnvClient` 调用：`/get_env_profile`（可选）、`/create`、`/step`、`/evaluate`、`/release`、`/healthz`。请求/响应结构见 `docs/guidelines/env_service.md` 与 `env_service/interface.md`。

---

## 第二层：Environment Profile（必须）

Self-Questioning 需要 **Environment Profile** 描述环境中的概念（实体、属性、操作）与任务偏好，供探索与总结阶段的 Prompt 使用。

### 2.1 JSON Profile（推荐）

在 `cookbook/env_profiles/` 下新增 `myenv.json`，结构示例：

```json
{
  "name": "用户角色名",
  "background": "用户背景描述",
  "entities": [
    {
      "name": "实体名",
      "description": "实体描述",
      "attrs": {
        "属性名": "属性说明"
      },
      "opts": [
        { "name": "操作名", "description": "操作说明" }
      ]
    }
  ],
  "task_preference": {
    "num_entities": 2,
    "num_opts": 3,
    "relation_difficulty": 3
  }
}
```

- **Entity**：环境中的核心对象（用户视角抽象，不必与每个 API 一一对应）。
- **Attribute**：实体的描述性元数据。
- **Operation**：可对实体执行的高层能力（可与真实 API 对齐，但不必穷举底层参数）。
- **task_preference**：控制生成任务的规模与难度；`relation_difficulty` 为 1–3（易 / 中 / 难，含跨实体与依赖关系）。

可参考 `cookbook/env_profiles/appworld.json`、`bfcl.json`、`webshop.json`。

### 2.2 代码方式（可选）

使用 `agentevolver.module.task_manager.env_profiles` 中的 `EnvProfile`、`EnvEntity`、`EnvEntityOpt`、`TaskPreference` 构建对象，并可 `save_to_json` / `load_from_json`。详见 [Task Manager 指南](task_manager.md)。

### 2.3 Rubrics（可选）

若需对生成任务施加额外约束（领域规则、禁止类任务等），可在代码中对 `EnvProfile` 调用 `reg_rubric(...)`；内容会进入总结阶段的任务偏好说明（`get_task_preference_instruction()`）。

---

## 第三层：训练与 Task Manager 配置（必须）

在 Hydra 配置中设置 `task_manager`（可与 `examples/`、`config/agentevolver.yaml` 对齐）。

### 3.1 常用字段示例

```yaml
task_manager:
  train_data_path: myenv_tasks.train.json   # 可选；不设则在集成模式下可动态生成
  val_data_path: myenv_tasks.val.json
  llm_client: qwen-plus
  n: 3                                      # 每个种子任务重复探索轮数
  env_profile: cookbook/env_profiles/myenv.json
  bs: ${data.train_batch_size}
  num_explore_threads: 16

  mixture:
    use_original_tasks: True
    synthetic_data_ratio: 0.5
    shuffle: True

  grader:
    original_grader: env                    # 环境自带评判
    synthetic_grader: llm                   # 合成任务常用 LLM 评判

  strategy: random
  strategy_args:
    max_explore_step: 30
    max_llm_retries: 6
    env_url: ${env_service.env_url}
    exploration_llm_temperature: 1.0
    exploration_llm_top_p: 1.0
    exploration_llm_top_k: 100
```

### 3.2 调参要点

| 参数 | 说明 |
|------|------|
| `max_explore_step` | 单次探索最大步数；环境简单时可适当减小 |
| `n` | 越大越多样，成本越高 |
| `exploration_llm_temperature` | 较高有利于探索多样性 |
| `synthetic_data_ratio` | 建议先 `0.0` 跑通原始任务，再逐步提高合成比例 |
| `train_data_path` / `val_data_path` | 设置后会复用/落盘探索结果；集成模式下不设路径则可能每轮动态生成（见 task_manager 文档） |

---

## 第四层：按需适配（可选）

### 4.1 探索与总结 Prompt

- **探索**：`agentevolver/module/task_manager/strategies/common/prompts/prompt_explore.py`  
  通过占位符插入 `EnvProfile.get_instruction()`；交互形态与 AppWorld 类 API 差异很大时再改。
- **总结**：`.../prompt_summarize.py`  
  依赖轨迹中 `assistant` 后紧跟 `tool` 或 `user` 的 observation 配对；若 observation 形态非文本或结构特殊，可能需要调整 `_get_action_observation_pair` 或 Prompt。

### 4.2 过滤器

- `filters/filters.py`（`NaiveTaskPostFilter`）：基于词集合的相似度去重，可按任务长度调整阈值逻辑。
- `filters/llm_filter.py`（`LlmFilter`）：会实际 rollout 验证任务；环境昂贵时可暂时关闭或弱化。

### 4.3 探索策略

当前内置 `strategy: random` → `LlmRandomSamplingExploreStrategy`。新策略需实现 `TaskExploreStrategy`（`explore` / `summarize`），并在 `task_manager.py` 的 `get_exploration_strategy` 中注册名称。

### 4.4 评判器

- 原始任务：环境实现 `evaluate` 且配置 `original_grader: env`。
- 合成任务：常用 `synthetic_grader: llm`，以参考解法（`action_sequence`）等与轨迹对比打分。

---

## 相关代码与文档路径

| 内容 | 路径 |
|------|------|
| Task Manager 主逻辑 | `agentevolver/module/task_manager/task_manager.py` |
| Random 探索策略 | `agentevolver/module/task_manager/strategies/random/__init__.py` |
| Env Profile 类型 | `agentevolver/module/task_manager/env_profiles.py` |
| 数据混合 | `agentevolver/module/task_manager/data_mixture.py` |
| 任务 → RL 数据集 | `agentevolver/module/task_manager/adapter.py` |
| Task / TaskObjective | `agentevolver/schema/task.py` |
| Env 基类 | `env_service/base.py` |
| Task Manager 详细说明 | [task_manager.md](task_manager.md) |
| Env Service 说明 | [env_service.md](env_service.md) |

---

## 推荐验证顺序

1. **Env Service**：启动服务，用 HTTP 验证 `/create` → `/step` → `/evaluate` → `/release`。
2. **Task Manager 单机**：`python -m agentevolver.module.task_manager`（需按项目实际入口与配置），检查合成任务 JSON 是否合理。
3. **迭代 Profile 与 YAML**：调整实体、操作、`task_preference` 与 `strategy_args`，直到任务质量可接受。
4. **接入完整训练**：通过 `launcher.py` 或 `main_ppo` 与 `env_service.env_url`、数据路径一并联调。

---

## 迁移清单速查

**必须完成**

- [ ] 实现 `BaseEnv` 并注册到 `env_service/environments/<name>/`
- [ ] 编写 `cookbook/env_profiles/<name>.json`（或等价 `EnvProfile`）
- [ ] 添加环境启动脚本（`env_service/launch_script/`）
- [ ] 配置 Hydra：`task_manager`、`env_service.env_url`、`grader`、`mixture`、`strategy_args`

**视场景调整**

- [ ] 自定义探索/总结 Prompt 或轨迹解析
- [ ] 调整去重/可行性过滤策略
- [ ] 注册新的 `TaskExploreStrategy`
- [ ] 自定义或对齐评判逻辑（env / llm）

Profile 质量直接决定合成任务是否贴近真实需求，建议在独立 Task Manager 阶段多轮迭代后再大规模训练。
