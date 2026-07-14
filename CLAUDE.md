# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / Run / Test Commands

### Install
```bash
python -m pip install -r requirements.txt
# Dev dependencies (from Dockerfile): pytest, ruff, pre-commit
python -m pip install pytest ruff pre-commit
```

### Dataset generation
```bash
python build_dataset.py --num_samples 2000 --output_dir ./data --model_path /path/to/model \
  --rewrite_backend vllm --vllm_gpu_mem_util 0.8 --vllm_max_model_len 8192
```
Quick smoke test (script-style helper, not pytest):
```bash
python tests/test_template_dataset.py --num_samples 10 --output_dir ./data/template_test --model_path /path/to/model
```

### Start the vLLM HTTP service (OpenAI-compatible endpoint)
```bash
bash vllm_service_init/start.sh /path/to/model run_id 0 5000
# Env knobs: VLLM_GPU_MEM_UTIL=0.8, VLLM_MAX_MODEL_LEN=8192
```
The service exposes `POST /v1/chat/completions` returning `{choices: [{message: {content: "..."}}]}`.

### Tests
```bash
python -m pytest -q                                        # all tests
python -m pytest -q tests/test_reward.py::test_reward_helpful_no_leak  # single test
```
Some files under `tests/` are script-style smoke checks (run with `python`, not pytest):
```bash
python tests/test_leakage_reward.py
python tests/test_rewrite_prompts.py --model_path /path/to/model --rewrite_backend vllm
python tests/test_defender_agent.py --model_path /path/to/model --port 5010

# PrivacyLens sandbox tests
python tests/test_privacylens_sandbox.py --mock                    # 71 mock tests (no GPU)
python tests/run_pl_sandbox_e2e.py --port 5000 --max_cases 6       # Real sandbox E2E
```

### Lint
```bash
ruff check .
```

### Iterative training
Docs: `ITERATIVE_TRAINING.md`. The iterative loop (`training/iterative_train_with_verl.sh`) regenerates the attack dataset each round using the latest attacker checkpoint, then runs RL training.
```bash
# Attacker training (starts guard vLLM, then runs iterative_train_with_verl.sh)
ATTACKER_GPU=0 GUARD_GPU=1 GUARD_PORT=5000 bash scripts/attacker_train.sh /path/to/attacker_ckpt attacker_run 0

# Top-level entry point
ATTACKER_GPU=0 GUARD_GPU=1 bash scripts/main.sh
```
Environment overrides: `ROUNDS`, `STEPS_PER_ROUND`, `SAMPLES_PER_ROUND`, `BASE_MODEL`, `ATTACKER_MODEL`, `GUARD_MODEL`, `WORKSPACE`, `PROJECT_DIR`.

## High-Level Architecture

This project implements a **self-evolving privacy RL** loop: an attacker model is trained via GRPO to generate adversarial prompts that trick a defender/guard model into leaking private information. Each training round regenerates the attack dataset from the latest attacker checkpoint, creating an adversarial co-evolution.

The project supports **three defender modes**:

### Core loop
```
Attacker (policy being trained)
    │ generates attack prompts
    ▼
Defender agent (guard model, served via vLLM)
    │ Standard mode: file-based ReAct agent
    │ PrivacyLens mode: ToolEmu 3-model sandbox (Agent + Simulator + Critiquer)
    │ AgentDAM mode: real browser agent (Playwright + web navigation)
    ▼
Reward = leakage_score + task_completion_bonus
    │ feeds back via GRPO
    ▼
Updated attacker → next round: regenerate dataset with new checkpoint
```

### Component map

**`attacker/` — Attack prompt generation**
- `state.py`: `AttackState` is the central data object with three dimensions:
  - **Targets** (what to extract): `profile`, `financial`, `medical`, `location`, `memory`
  - **Strategies** (how to attack): `direct`, `indirect`, `roleplay`, `reasoning`, `multi_turn`
  - **Paths** (where private data lives): `memory`, `tool`, `reasoning_path`, `trajectory_path`
  - **Task Types**: `file_edit`, `email_reply`
- `prompt_templates.py`: `PROMPT_BUILDERS` maps each strategy name to a function that generates attacker model prompts (meta-prompts instructing the attacker LLM to craft adversarial messages).
- `dataset_builder.py`: `build_attack_dataset()` generates train/val JSON files for standard mode. `build_pl_style_records()` generates PrivacyLens-mode records with `agent_type: "privacylens"`, including `privacy_sensitive_context` and `sensitive_info_items`.
- `scenario_generator.py`: `ScenarioGenerator` produces PrivacyLens-compatible scenarios (user instruction, privacy-sensitive context, toolkits, sensitive items). Used by `build_pl_style_records()`.
- `agent.py`: `process_attacker_response()` parses attacker model output for **standard mode** (splits into user_request + optional document_content for indirect attacks).

**`defender/` — Guard agent**
- `agent.py`:
  - `run_agent()` — **Standard mode**: file-based ReAct agent with `list_dir`/`read_file`/`write_file` tools. Secrets stored at `sandbox/secrets.txt`. Supports `task_file_content` and `document_content` for indirect attacks. Max 5 steps by default.
  - `run_pl_agent()` — **PrivacyLens mode**: uses the real PrivacyLens ToolEmu sandbox. The Agent LLM (guard model) navigates virtual services while a Simulator LLM dynamically generates tool-output Observations embedding sensitive data. Returns trajectory + final action.
- `memory.py`: `MemoryStore` handles writing secrets and public documents to disk (standard mode).
- `tools.py`: `ReadFileTool`, `ListDirTool`, `WriteFileTool` — filesystem operations (standard mode). Error results are prefixed with `[ERROR]`.
- `guard_model.py`: `call_guard_llm()` sends chat messages to the guard vLLM endpoint. Configured via `GUARD_OPENAI_URL`, `GUARD_MODEL`, `GUARD_TIMEOUT` env vars.
- `privacylens/` — **PrivacyLens ToolEmu sandbox integration**:
  - `sandbox.py`: `PrivacyLensSandbox` wraps PrivacyLens's three-model (Agent + Simulator + Critiquer) trajectory simulation. The Simulator LLM follows `privacy_sensitive_context` to generate adversarial tool outputs. Imports from `/home/fangzibang/PrivacyLens/data_construction/toolemu/`. Contains runtime patches for pydantic v1/v2 compatibility.
  - `vllm_adapter.py`: `VLLMChatModel` — LangChain-compatible **BaseLLM** (not BaseChatModel) that routes ToolEmu LLM calls to the local vLLM HTTP endpoint. Extends BaseLLM so PrivacyLens's agent builder treats it as non-chat format.
  - Requires **procoder** from `/home/fangzibang/PromptCoder/procoder/` (SALT-NLP prompt framework). Also requires `langchain==0.0.277`.
- `agentdam_defender.py` — **AgentDAM mode**: real browser agent using Playwright + VisualWebArena. Navigates actual websites (Reddit, Shopping, GitLab) and performs web actions. Requires Docker containers and Playwright.

**`reward/` — Reward computation**
- `interface.py`: `RewardInterface` does literal substring matching for leakage detection — splits sensitive items, extracts values, checks if any appear (case-insensitive) in the response.
  - `detect_leakage()`: Returns score in [0, 1] with word-level partial credit.
  - `compute_reward()`: Returns `{overall, leaked, task_completed}`.
- `verl_reward_wrapper.py`: `compute_score(data_items, **kwargs)` is the entry point called by veRL. Routes each item to the appropriate defender based on `agent_type`:
  - `"privacylens"` → runs `run_pl_agent()` with real ToolEmu sandbox
  - `"agentdam"` → runs `run_agentdam_defender_from_attacker()` with browser agent
  - default → runs `run_agent()` standard file-based agent
- `agentdam_reward.py`: AgentDAM-specific reward computation.

**`verl/` — Vendored RL training framework**
- Entry point: `python -m verl.trainer.main`.
- Config: OmegaConf/Hydra with YAML (`examples/config.yaml`).

### Data contract

**Standard mode** dataset records:
```json
{
  "prompt": "...", "target": "financial", "sensitive": "bank card tail: 1234; pin: 5678",
  "strategy": "direct", "task_type": "file_edit", "task_file": "...", "email": "..."
}
```

**PrivacyLens mode** dataset records:
```json
{
  "prompt": "...", "target": "medical_information", "sensitive": "",
  "sensitive_info_items": ["Dave was diagnosed with anxiety."],
  "strategy": "direct", "task_type": "send_email",
  "agent_type": "privacylens",
  "user_instruction": "Send an update to Susan...",
  "user_name": "Jane Doe", "user_email": "jane.doe@gmail.com",
  "privacy_sensitive_context": "Context: ...\nDetailed Plot: ...",
  "toolkits": ["Gmail", "NotionManager"], "final_action": "GmailSendEmail"
}
```

The reward function receives `response` (attacker's generated text) and `ground_truth`/`sensitive_info_items` (sensitive data). Dataset keys must stay aligned with `data.prompt_key=prompt` and `data.answer_key=sensitive` in veRL config.

### Service wiring
The guard model is always served as an HTTP service. The reward wrapper and defender agent both call it. Configuration via env vars:
- `GUARD_OPENAI_URL` — guard endpoint (default `http://127.0.0.1:8000/v1/chat/completions`)
- `GUARD_MODEL` — model name for the OpenAI-style payload
- `GUARD_TIMEOUT` — request timeout in seconds (default 30)
- `GUARD_USE_MOCK=true` — bypass all LLM calls, return mock responses
- `PL_SANDBOX_USE_MOCK=true` — bypass PrivacyLens sandbox, use mock trajectory
- `GUARD_TIMEOUT` — request timeout in seconds (default 30, use 120 for sandbox)

### Dependency notes
- **procoder**: Not on PyPI. Uses local copy at `/home/fangzibang/PromptCoder/procoder/`. Added to `sys.path` by `sandbox.py`. Requires `roman` package.
- **langchain==0.0.277**: Required by PrivacyLens ToolEmu. Must be exact version — newer langchain has breaking API changes.

Port/GPU assignments are standardized in `scripts/common.sh`:
- Guard: `GUARD_PORT` (default 5000), `GUARD_GPU` (default 2)
- Attacker vLLM: `ATTACKER_VLLM_PORT` (default 5001), `ATTACKER_GPU` (default 0)

### Path conventions
- Shell scripts default `WORKSPACE=/home/fangzibang` and derive `PROJECT_DIR=$WORKSPACE/SelfEvolvingPrivacyRL`.
- When making scripts portable, prefer reading these from env rather than hard-coding absolute paths.

### External dependencies (PrivacyLens sandbox mode)
The PrivacyLens sandbox requires two external codebases. Configure via environment variables:

| Env Var | Default | Description |
|---------|---------|-------------|
| `PRIVACYLENS_HOME` | `/home/fangzibang/PrivacyLens` | PrivacyLens repo (contains `toolemu/` ToolEmu framework) |
| `PROCODER_HOME` | `/home/fangzibang/PromptCoder` | PromptCoder repo (SALT-NLP procoder prompt framework) |

These are only needed at **runtime** when using `run_pl_agent()` → `PrivacyLensSandbox`.  
**Not needed** for dataset generation (`build_dataset.py`) or standard/AgentDAM modes.

- AgentDAM codebase is at `/home/fangzibang/ai-agent-privacy` (imported by `defender/agentdam_defender.py`).

### Reward function spec format
Use `path/to/file.py:function_name` (e.g., `reward/verl_reward_wrapper.py:compute_score`) when setting `worker.reward.reward_function` in veRL config.
