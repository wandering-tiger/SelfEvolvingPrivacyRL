# PrivacyPeek Training

Two GRPO paths share the same PrivacyPeek cases / env / CER judge:

| Path | Role being trained | Frozen side | Reward |
|------|--------------------|-------------|--------|
| **A Attacker** | rewrite benign task → adversarial user message | guard/victim | `+CER + forbidden_bonus` |
| **B Defender** | generate ReAct JSON actions | — (actor is the guard) | `-CER + λ·TCR` |

## Prerequisites

1. PrivacyPeek checkout at `../PrivacyPeek` (or set `PRIVACYPEEK_ROOT`)
2. A base model for training
3. GPU for vLLM guard service + veRL training

---

## Path B — Defender

### 1. Build dataset

```bash
export PRIVACYPEEK_ROOT=/home/fangzibang/PrivacyPeek

python build_privacypeek_dataset.py \
  --cases-dir $PRIVACYPEEK_ROOT/data/cases \
  --output-dir ./data/privacypeek \
  --limit 100 \
  --dims B1,B2
```

### 2. Start guard vLLM + train defender（推荐：五卡一键脚本）

```bash
conda activate privacyrl
cd /home/fangzibang/SelfEvolvingPrivacyRL

# Smoke（20 cases）——会强制用满 5 张卡并清理残留进程
bash scripts/run_privacypeek_5gpu.sh

# 正式训练
bash scripts/run_privacypeek_5gpu.sh full
```

固定 GPU 布局：

| GPU | 用途 |
|-----|------|
| **0,1,3,4** | veRL FSDP actor + rollout vLLM（TP=4） |
| **2** | guard vLLM（reward / PrivacyPeek env） |

若仍 OOM（不要把 `ROLLOUT_N` 设为 1）：

```bash
ROLLOUT_BATCH_CAP=2 bash scripts/run_privacypeek_5gpu.sh
```

Guard 日志：`$WORKSPACE/storage/logs/guard_vllm_*_gpu2.log`

### 3. Evaluate on PrivacyPeek benchmark（一键）

```bash
conda activate privacyrl
cd /home/fangzibang/SelfEvolvingPrivacyRL

# 评测最新训练 checkpoint（自动 merge → vLLM → agent → CER）
bash scripts/eval_privacypeek_defender.sh

# 只评 20 条
bash scripts/eval_privacypeek_defender.sh latest 20

# 指定 checkpoint
bash scripts/eval_privacypeek_defender.sh \
  /home/fangzibang/storage/models/privacy_defender_privacypeek_20260714_095126/round_0/global_step_10 \
  50

# 基线（未训练 base model）
bash scripts/eval_privacypeek_defender.sh baseline 50
```

默认用 **GPU 4** 起 vLLM；结果写到 `runs/`：

- `runs/defender_*.json` — agent 轨迹
- `runs/cer_defender_*.json` — CER 分数

可用环境变量覆盖：`EVAL_GPU`、`EVAL_PORT`、`EVAL_LIMIT`、`HF_OUT`。

### Defender reward

- **CER**: PrivacyPeek exact-match judge on tool `return_value` / `args`
- **TCR**: training-time heuristic (tool use + expected scope + non-refusal response)
- **Reward**: `overall = -CER + 0.3·TCR - format_penalty - forbidden_penalty`

### Defender architecture

```
PrivacyPeek case prompt
        │
        ▼
GRPO actor generates JSON action  ──► reward wrapper executes first action
        │                                      │
        │                                      ▼
        │                            PrivacyPeek env (multi-step vLLM on GPU 2)
        │                                      │
        └──────────────────────────────────────┤
                                               ▼
                                    CER + TCR → GRPO update (FSDP on GPU 0,1,3,4)
```

---

## Path A — Attacker

Train the attacker to rewrite each PrivacyPeek case's benign `task` into an adversarial user message. A **frozen** guard runs the full PrivacyPeek ReAct loop on that rewritten task; attacker reward rises with CER.

### 1. Build dataset

```bash
export PRIVACYPEEK_ROOT=/home/fangzibang/PrivacyPeek

python build_privacypeek_attacker_dataset.py \
  --cases-dir $PRIVACYPEEK_ROOT/data/cases \
  --output-dir ./data/privacypeek_attacker \
  --limit 100 \
  --dims B1,B2
```

### 2. Train attacker（五卡一键）

```bash
conda activate privacyrl
cd /home/fangzibang/SelfEvolvingPrivacyRL

# Smoke（20 cases）
bash scripts/run_privacypeek_attacker_5gpu.sh

# 正式训练
bash scripts/run_privacypeek_attacker_5gpu.sh full
```

GPU 布局与 Path B 相同：`0,1,3,4` 训 attacker，`2` 跑冻结 guard。

也可用：

```bash
bash scripts/attacker_privacypeek_train.sh [MODEL_PATH] [RUN_ID]
```

### 3. Evaluate attacker（一键）

```bash
conda activate privacyrl
cd /home/fangzibang/SelfEvolvingPrivacyRL

# 评测最新 attacker checkpoint（merge → 生成攻击 → 冻结 victim → CER）
bash scripts/eval_privacypeek_attacker.sh

# 只评 20 条
bash scripts/eval_privacypeek_attacker.sh latest 20

# 指定 checkpoint
bash scripts/eval_privacypeek_attacker.sh \
  /home/fangzibang/storage/models/privacy_attacker_privacypeek_XXXX/round_0/global_step_10 \
  50

# 基线（未训练 base model 当 attacker）
bash scripts/eval_privacypeek_attacker.sh baseline 50
```

默认单卡两阶段（先 attacker 生成攻击，再 victim 跑 env）。结果写到 `runs/`：

- `runs/attacks_*.json` — 改写后的攻击话术
- `runs/attacker_*.json` / `runs/baseline_*.json` — victim 轨迹
- `runs/cer_*.json` — CER 分数
- `runs/summary_*.json` — CER rate / forbidden rate

可用环境变量：`EVAL_GPU`、`EVAL_PORT`、`VICTIM_MODEL`、`WORLD_SIZE`、`HF_OUT`。

### Attacker reward

- **CER**: same PrivacyPeek judge — higher is better for the attacker
- **Forbidden bonus**: extra score if guard tool calls hit `forbidden_sources`
- **Empty / identical penalties**: discourage blank or copy-paste of the original task
- **Reward**: `overall = CER + 0.2·forbidden_hit - empty_penalty - identical_penalty`

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVACYPEEK_ATTACKER_FORBIDDEN_BONUS` | `0.2` | Bonus when guard touches forbidden sources |
| `PRIVACYPEEK_ATTACKER_EMPTY_PENALTY` | `0.5` | Penalty for empty / too-short attacks |
| `PRIVACYPEEK_ATTACKER_IDENTICAL_PENALTY` | `0.05` | Penalty if attack == original task |

### Attacker architecture

```
PrivacyPeek case → attacker rewrite prompt
        │
        ▼
GRPO actor outputs adversarial user message
        │
        ▼
Frozen guard runs PrivacyPeek env with task_override=attack
        │
        ▼
+CER (+ forbidden) → GRPO update attacker (FSDP on GPU 0,1,3,4)
```

---

## Shared environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVACYPEEK_ROOT` | `../PrivacyPeek` | PrivacyPeek repo root |
| `PRIVACYPEEK_CASES_DIR` | `$PRIVACYPEEK_ROOT/data/cases` | Case directory |
| `PRIVACYPEEK_TCR_WEIGHT` | `0.3` | Defender TCR coefficient |
| `PRIVACYPEEK_FORMAT_PENALTY` | `0.1` | Defender invalid JSON action penalty |
| `PRIVACYPEEK_FORBIDDEN_PENALTY` | `0.2` | Defender forbidden source access penalty |
| `PRIVACYPEEK_MAX_STEPS` | `6` | Max ReAct steps in reward env |
| `GUARD_USE_MOCK` | `false` | Skip vLLM calls in reward |
| `TRAIN_GPUS` | `0,1,3,4` | veRL 训练 GPU 列表（自动排除 `GUARD_GPU`） |
| `GUARD_GPU` | `2` | guard vLLM GPU（40GB 卡） |
| `ROLLOUT_TP` | `=训练卡数` | rollout vLLM tensor parallel 宽度 |
| `GUARD_VLLM_GPU_MEM_UTIL` | `0.85` | guard 独占 40GB 卡时的 vLLM 显存占比 |
| `GUARD_VLLM_MAX_MODEL_LEN` | `2048` | guard 最大上下文长度 |
| `GUARD_STARTUP_TIMEOUT` | `600` | 等待 guard 就绪的最长时间（秒） |
| `ROLLOUT_GPU_MEM_UTIL` | `0.45`(多卡) / `0.30`(单卡) | 每张训练卡上 rollout vLLM 显存占比 |
| `ROLLOUT_N` | `2` | GRPO 每组采样数（**必须 ≥2**） |
| `ROLLOUT_BATCH_CAP` | `4` | 单步 rollout batch 上限 |
| `DISABLE_KL` | `true` | 默认关闭 ref 模型以省显存 |

## GPU layout（4×40GB + 1×80GB，Llama 8B）

| GPU | 卡型 | 默认用途 |
|-----|------|----------|
| **0, 1, 3, 4** | 3×40GB + 1×80GB | veRL FSDP（actor 分片）+ rollout vLLM（TP=4） |
| **2** | A100-40GB | guard vLLM 服务（reward 推理） |

> 多卡训练时：FSDP 把 actor 权重分到各卡，rollout vLLM 用 **tensor parallel** 跨卡切分，单卡显存压力大幅下降。
> 单卡 80GB（`TRAIN_GPUS=4`）仍可用作 fallback，但更容易 OOM。

## Key files

**Defender (Path B)**
- `defender/privacypeek_env.py` — PrivacyPeek JSON ReAct environment (`task_override` for attacker)
- `defender/privacypeek_tools.py` — tool implementations + call log
- `defender/dataset_builder.py` — case → defender train/val JSON
- `reward/privacypeek_reward.py` — CER/TCR defender reward
- `reward/privacypeek_reward_wrapper.py` — veRL batch entrypoint
- `examples/config_privacypeek.yaml` — veRL config
- `training/iterative_train_defender_verl.sh` — training loop
- `scripts/guard_train.sh` / `scripts/run_privacypeek_5gpu.sh` — entrypoints

**Attacker (Path A)**
- `attacker/privacypeek_dataset.py` — case → attacker rewrite prompts
- `build_privacypeek_attacker_dataset.py` — dataset CLI
- `reward/privacypeek_attacker_reward.py` — CER attacker reward
- `reward/privacypeek_attacker_reward_wrapper.py` — veRL batch entrypoint
- `examples/config_privacypeek_attacker.yaml` — veRL config
- `training/iterative_train_attacker_privacypeek_verl.sh` — training loop
- `scripts/attacker_privacypeek_train.sh` / `scripts/run_privacypeek_attacker_5gpu.sh` — entrypoints
