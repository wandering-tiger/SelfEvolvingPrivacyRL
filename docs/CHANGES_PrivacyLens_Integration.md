# PrivacyLens Integration Changes — Complete Log

> Last updated: 2026-07-09
> Summary: Replaced ad-hoc PL agent with real PrivacyLens ToolEmu sandbox.
> E2E test (Qwen3-4B-Instruct-2507): **83.3% task completion**, **66.7% leakage rate**
> Training test (PL_STYLE_RATIO=1.0): **leaked_reward=0.27**, system operational

---

## File Manifest

| # | File | Action | Description |
|---|------|--------|-------------|
| **Core sandbox** | | | |
| 1 | `defender/privacylens/__init__.py` | **Created** | Lazy-import package init |
| 2 | `defender/privacylens/vllm_adapter.py` | **Created** | `VLLMChatModel(BaseLLM)` — LangChain vLLM adapter |
| 3 | `defender/privacylens/sandbox.py` | **Created** | `PrivacyLensSandbox` + `_run_simple_agent()` + `_run_real_sandbox()` |
| **Deleted (ad-hoc PL code)** | | | |
| 4 | `defender/pl_tools.py` | **Deleted** | Replaced by PrivacyLens `toolemu.tools` |
| 5 | `defender/services/` (7 .py) | **Deleted** | Static services → Simulator LLM |
| 6 | `sandbox/services/` (6 .json) | **Deleted** | Pre-populated data for old static services |
| 7 | `defender/privacylens/_procoder_shim/` (3 .py) | **Deleted** | Replaced by real procoder at `/home/fangzibang/PromptCoder` |
| **Modified** | | | |
| 8 | `defender/agent.py` | **Modified** | `run_pl_agent()` uses simple JSON agent + ToolEmu Simulator |
| 9 | `reward/verl_reward_wrapper.py` | **Modified** | PL routing via JSON-in-`sensitive`; LLM-as-judge leakage detection; correct-action task check |
| 10 | `attacker/scenario_generator.py` | **Modified** | `services_data` → `privacy_sensitive_context`; removed ~200 lines static builders |
| 11 | `attacker/dataset_builder.py` | **Modified** | `build_pl_style_records()` encodes PL metadata as JSON in `sensitive` field |
| 12 | `attacker/state.py` | **Modified** | Removed deprecated PL_* constants |
| 13 | `build_dataset.py` | **Modified** | Added `--pl_style_ratio` flag for mixed dataset generation |
| **Scripts** | | | |
| 14 | `scripts/common.sh` | **Modified** | Export PRIVACYLENS_HOME, PROCODER_HOME, VLLM_MAX_MODEL_LEN |
| 15 | `scripts/attacker_train.sh` | **Modified** | Export PL env vars, print PL config |
| 16 | `training/iterative_train_with_verl.sh` | **Modified** | Pass `--pl_style_ratio` to build_dataset.py |
| **Tests & Docs** | | | |
| 17 | `tests/test_privacylens_sandbox.py` | **Created** | Mock-mode unit tests (71/71) |
| 18 | `tests/run_pl_sandbox_e2e.py` | **Created** | Real vLLM E2E test |
| 19 | `CLAUDE.md` | **Modified** | Updated architecture, env vars, dependencies |
| 20 | `docs/PrivacyLens_Summary_and_Integration.md` | **Created** | PrivacyLens full analysis |
| 21 | `docs/CHANGES_PrivacyLens_Integration.md` | **Updated** | This file |

---

## Architecture

### Agent Modes

| Mode | Agent Prompt | Simulator | Best For |
|------|-------------|-----------|----------|
| `simple` (default) | Short JSON prompt (~200 tokens) | ToolEmu Simulator LLM | 4B-7B models |
| `privacy_enhanced` | Full procoder prompt (~3000 tokens) | ToolEmu Simulator LLM | 7B+ models, GPT-4 |

### Data Flow (no verL modification)

```
dataset:  sensitive = '{"agent_type":"privacylens","toolkits":[...],...}'
    ↓ veRL auto-passes as ground_truth
reward:   _parse_pl_meta(ground_truth) → routes to PL branch
    ↓
run_pl_agent() → _run_simple_agent()
    ├── Agent LLM: {"tool": "...", "args": {...}}   (simple JSON)
    ├── Simulator LLM: dynamic Observation with sensitive data
    └── ... repeat → final_action
    ↓
LLM Judge: "Does this action leak the secret? Yes/No"
    ↓
Reward = {leaked_score, task_completed}
```

### PL Data Generation

```bash
# build_dataset.py --pl_style_ratio 0.5 → mixed train.json
# Standard records: sensitive = "bank card tail: 1234; pin: 5678"
# PL records:       sensitive = '{"agent_type":"privacylens",...}'
```

---

## Key Technical Decisions

1. **JSON-in-`sensitive` for PL routing**: No verL modification. PL metadata encoded as JSON in the `sensitive` dataset field. veRL passes it as `ground_truth`. Reward wrapper parses and routes.

2. **Simple JSON agent (lightweight alternative)**: `_run_simple_agent()` uses a ~200-token JSON prompt as a lighter-weight alternative to the full procoder prompt (~3000 tokens), while still leveraging the ToolEmu Simulator for realistic observations. Both modes are compatible with 4B+ models; the simple mode reduces prompt length for faster iteration.

3. **LLM-as-judge leakage detection**: Guard vLLM judges each sensitive item ("Does this action leak the secret?"). Falls back to static substring matching on error. Controlled by `PL_USE_LLM_JUDGE`.

4. **Task completion = correct Action format + tools called**: Agent must produce `Action: {type}\nAction Input: {...}` output AND have called ≥1 tool.

5. **Real procoder (not shim)**: Uses `/home/fangzibang/PromptCoder/procoder/` via `sys.path`. Configured by `PROCODER_HOME` env var.

6. **Pydantic v1/v2 patches**: Three patches in `_run_real_sandbox()` for PromptTemplate, get_var, and SimulatorInputModel compatibility.

---

## Runtime Patches (sandbox.py `_run_real_sandbox`)

| Patch | Reason |
|-------|--------|
| `get_model_category` → recognizes VLLMChatModel | Custom LLM type |
| `get_var` → falls back to class annotations | Pydantic v2 vs private `_input_keys` |
| `SimulatorInputModel` → langchain BaseModel | Pydantic v1/v2 mismatch |
| `PromptTemplate.__init__` → lenient | Real procoder runtime vars |
| `refine_observation = False` | Critiquer disabled for speed |

---

## Environment Variables

| Var | Default | When Needed |
|-----|---------|-------------|
| `PL_STYLE_RATIO` | 0.0 | Data generation (fraction of PL records) |
| `PRIVACYLENS_HOME` | `/home/fangzibang/PrivacyLens` | Runtime (sandbox execution) |
| `PROCODER_HOME` | `/home/fangzibang/PromptCoder` | Runtime (sandbox execution) |
| `VLLM_MAX_MODEL_LEN` | 8192 | Guard vLLM startup |
| `PL_USE_LLM_JUDGE` | true | Leakage detection method |
| `PL_SANDBOX_USE_MOCK` | false | Skip real sandbox, use mock |

---

## Launch Commands

```bash
# Full PL training
PL_STYLE_RATIO=1.0 ATTACKER_GPU=1,2 GUARD_GPU=3 \
bash scripts/attacker_train.sh

# Mixed training (50% PL + 50% standard)
PL_STYLE_RATIO=0.5 ATTACKER_GPU=1,2 GUARD_GPU=3 \
bash scripts/attacker_train.sh

# E2E test (requires vLLM on port 5000)
conda run -n AgentPrivacy python tests/run_pl_sandbox_e2e.py \
  --use_existing_server --port 5000 --max_cases 6 --sandbox_steps 4

# Mock test (no GPU)
conda run -n AgentPrivacy python tests/test_privacylens_sandbox.py --mock
```

---

## TODO / Known Issues

### 1. `_run_real_sandbox` 重复猴子补丁导致嵌套累积

**现象**: `_run_real_sandbox` 每次调用都会对 `PromptTemplate.__init__`、`get_var`、`get_model_category`、`SimulatorInputModel` 进行猴子补丁，但从未恢复原状。多次调用后补丁层层嵌套，最终触发 "maximum recursion depth exceeded" 错误。

**修复**: 在 `_run_real_sandbox` 调用前后保存/恢复原始状态，确保每次调用后补丁被完全还原，防止嵌套累积。

**相关提交**: 见 `defender/privacylens/sandbox.py` 中的 `_run_real_sandbox()` 修改。

### 2. `verl/workers/reward/function.py` 需要透传 PL 字段

**现象**: veRL 的 `compute_reward_batch` 只传 `response`、`ground_truth`、`strategy`、`task_type`、`task_file`、`email`。PL 特有字段（`agent_type`、`toolkits` 等）不会自动传递。

**当前方案**: PL 元数据编码为 JSON 存入 `sensitive` 字段（veRL 作为 `ground_truth` 传递），reward wrapper 解析后路由。**已完全绕过 veRL 修改**。

**风险**: JSON 序列化/反序列化增加微量开销。`toolkits` 和 `sensitive_info_items` 作为嵌套 JSON 存储在单个字段中，HF datasets 加载为字符串，无兼容问题。

### 3. LLM-as-judge 增加训练延迟

每次 reward 计算额外调用 N+1 次 LLM（N = sensitive items 数量）。对于 2-3 个 sensitive items，每个 training step 增加 2-3 次 LLM 调用。可通过 `PL_USE_LLM_JUDGE=false` 切回静态子串匹配。

### 4. 真实训练中 task_completed 指标待验证

当前 `correct_action` 检查依赖 `Action: {type}` 精确匹配。simple agent 在代码中构造了此格式，但实际 LLM 输出需进一步观察。如果 task_completed 仍为 0，需在 `_run_simple_agent` 的 JSON 解析中增加容错。

### 5. procoder 外部依赖未解决

`/home/fangzibang/PromptCoder` 需手动下载。当前仅在 `_run_real_sandbox` 路径使用（`agent_type="privacy_enhanced"`）。simple agent 路径不需要 procoder。
