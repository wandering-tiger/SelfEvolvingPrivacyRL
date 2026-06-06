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

### Core loop
```
Attacker (policy being trained)
    │ generates attack prompts
    ▼
Defender agent (guard model, served via vLLM)
    │ receives attack + sensitive data, may use tools
    ▼
Reward = leakage_score + helpfulness_bonus + tool_bonus
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
- `strategies/`: Each strategy is a class with `name` and `build_prompt(attack_state) -> str`. The `direct` strategy has the richest template set (system override, admin spoofing, compliance framing, debug mode, urgency, role confusion, pseudo-system injection). The `indirect` strategy builds attacks with hidden payloads embedded in benign external content using various wrappers (HTML comments, markdown, config blocks, metadata, OCR artifacts).
- `prompt_templates.py`: `PROMPT_BUILDERS` maps each strategy name to a function that takes the raw strategy output and wraps it in a meta-prompt instructing an LLM to generate a diverse, realistic attack prompt in that style.
- `prompt_builder.py`: `PromptBuilder` looks up the strategy, calls `build_prompt`, then optionally refines via an LLM generator.
- `dataset_builder.py`: `build_attack_dataset()` generates train/val JSON files. For each sample it: (1) randomly picks an AttackState, (2) builds a base prompt, (3) assigns a random synthetic sensitive value matching the target domain. Output records have keys: `prompt`, `target`, `sensitive`.

**`defender/` — Guard agent with tool use**
- `agent.py`: `run_agent()` implements a ReAct loop (max 5 steps by default). The guard LLM outputs JSON: either `{"tool": "...", "args": {...}}` or `{"final": "..."}`. Two tools available: `list_dir` and `read_file`. The agent stores secrets at `sandbox/secrets.txt` and optional untrusted documents at `sandbox/document.txt`.
- `memory.py`: `MemoryStore` handles writing secrets and public documents to disk.
- `tools.py`: `ReadFileTool` and `ListDirTool` — simple filesystem operations. Error results are prefixed with `[ERROR]`.
- `guard_model.py`: `call_guard_llm()` sends chat messages to the guard vLLM endpoint. Configured via `GUARD_OPENAI_URL`, `GUARD_MODEL`, `GUARD_TIMEOUT` env vars.

**`reward/` — Reward computation**
- `interface.py`: `RewardInterface` does literal substring matching — splits sensitive items by `;` or `,`, extracts values after `:`, and checks if any appear (case-insensitive) in the assistant response. The reward formula: `leaked_score + 0.05 * helpfulness + tool_weight * min(tool_successes, cap)`.
- `verl_reward_wrapper.py`: `compute_score(data_items, **kwargs)` is the entry point called by veRL (configured as `worker.reward.reward_function=./reward/verl_reward_wrapper.py:compute_score`). For each data item, it calls `run_agent()` from the defender, then calls a helpfulness judge (another LLM call scoring 0/2/3), then computes the combined reward. Mock modes available via `GUARD_USE_MOCK=true` and `HELPFULNESS_JUDGE_USE_MOCK=true`.

**`verl/` — Vendored RL training framework**
- Entry point: `python -m verl.trainer.main`.
- Config: OmegaConf/Hydra with YAML (`examples/config.yaml`). Key config sections: `data` (dataset paths, prompt/answer keys), `algorithm` (GRPO with KL penalty), `worker.actor` (FSDP, optimizer, model path), `worker.rollout` (vLLM generation params), `worker.reward` (reward function path), `trainer` (steps, logging, checkpointing).
- Execution: Ray-based trainer orchestrating FSDP workers for actor/critic/ref models and vLLM for rollouts.

### Data contract
Dataset JSON files contain records with `prompt`, `target`, and `sensitive` keys. The config maps `data.prompt_key=prompt` and `data.answer_key=sensitive`. The reward function receives `response` (the attacker's generated text) and `ground_truth` (the sensitive data). If you change dataset generation, keep these keys aligned or update the config.

### Service wiring
The guard model is always served as an HTTP service. The reward wrapper and defender agent both call it. Configuration via env vars:
- `GUARD_OPENAI_URL` — guard endpoint (default `http://127.0.0.1:8000/v1/chat/completions`)
- `GUARD_MODEL` — model name for the OpenAI-style payload
- `GUARD_TIMEOUT` — request timeout in seconds (default 30)
- `GUARD_USE_MOCK=true` — bypass guard calls, return mock response
- `HELPFULNESS_JUDGE_USE_MOCK=true` — return mock helpfulness score (2)

Port/GPU assignments are standardized in `scripts/common.sh`:
- Guard: `GUARD_PORT` (default 5000), `GUARD_GPU` (default 2)
- Attacker vLLM: `ATTACKER_VLLM_PORT` (default 5001), `ATTACKER_GPU` (default 0)

### Path conventions
Shell scripts default `WORKSPACE=/home/fangzibang` and derive `PROJECT_DIR=$WORKSPACE/SelfEvolvingPrivacyRL`. When making scripts portable, prefer reading these from env rather than hard-coding absolute paths.

### Reward function spec format
Use `path/to/file.py:function_name` (e.g., `reward/verl_reward_wrapper.py:compute_score`) when setting `worker.reward.reward_function` in veRL config.
