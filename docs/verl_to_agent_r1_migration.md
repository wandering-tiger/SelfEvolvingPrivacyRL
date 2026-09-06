# 训练框架迁移指南：verl（vendored）→ Agent-R1

> 目标：把本项目的 GRPO 对抗训练（攻击者生成攻击提示 → defender 执行 → 泄密/任务奖励）从仓库内 vendored 的 verl 迁移到 `/home/fangzibang/Agent-R1`。
>
> 结论先行：**核心训练循环（单步生成 + 自定义 reward + 迭代式 checkpoint 加载）迁移成本很低**——不需要写 AgentFlow/工具环境，Agent-R1 内置的 `single_step_agent` 流程 + `custom_reward_function` 即可 1:1 复刻现有训练。需要改动的只有三处：**数据格式（JSON → parquet）、reward 函数签名（批量 → 单样本）、启动脚本（配置键全部变化）**。

---

## 1. 现状速览（迁移前）

| 组件 | 现状 |
|---|---|
| 训练入口 | `python3 -m verl.trainer.main config=examples/config.yaml`（仓库根目录 vendored `verl/` 包） |
| 配置 | 自定义 `worker.*` 键（`worker.actor.model.model_path`、`worker.reward.reward_function` 等），Hydra override |
| Reward 接口 | **批量式** `compute_score(reward_inputs: List[Dict]) -> List[Dict]`（`reward/verl_reward_wrapper.py`） |
| 数据集 | JSON list，`prompt`（字符串）+ `sensitive`（answer_key）+ 元数据字段（strategy/task_type/agent_type/...） |
| 迭代训练 | `training/iterative_train_with_verl.sh`：每轮用最新 attacker ckpt 重新生成数据 → verl 训练，`trainer.load_checkpoint_path=.../global_step_*` 加载上轮 ckpt |
| Defender | 独立 vLLM HTTP 服务（`GUARD_OPENAI_URL`），reward 函数内部通过 HTTP 调 defender 跑轨迹并算泄密/任务奖励 |
| 运行环境 | conda `AgentPrivacy` 环境（Python 3.10.20，vllm 0.11.0，ray 2.55.1）；底层依赖齐备，**verl 包本身和 agent_r1 尚未安装**（只需补装 `verl==0.7.0`） |

## 2. Agent-R1 是什么

- **基于 verl 0.7.0**（pip 依赖，不是 fork、不是子模块）：`agent_r1` 包在 verl 之上做 step-level MDP 重构（`RayAgentTrainer`、`AgentFlowBase` 体系、trajectory mini-batch、`AgentAdvantageEstimator`）。
- **入口**：`python3 -m agent_r1.trainer.main_agent_ppo` + 全部 Hydra overrides（无 config 文件，主配置 `agent_r1/config/agent_ppo_trainer.yaml` 继承 verl 0.7.0 的 `ppo_trainer.yaml`）。
- **reward 接口**：单样本式 `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`。
- **数据**：parquet（`data.return_raw_chat=True` 时 `prompt` 为 chat messages 列表）。
- **checkpoint**：`trainer.resume_mode`（`auto` / `disable` / `resume_path`）+ `trainer.resume_from_path`（**路径必须含 `global_step_`**）。
- **GRPO**：`algorithm.adv_estimator=grpo` + `critic.enable=False` + `reward_model.enable=False`，`actor_rollout_ref.rollout.n` 控制每组采样数。

参考示例（Agent-R1 仓库内）：`examples/hotpotqa/run_grpo.sh`（GRPO 最接近本项目的参考）、`examples/gsm8k/run_steppo.sh`（最小单步示例）、`recipes/gsm8k/reward_fn.py`（自定义 reward 最小示例）。

## 3. 核心差异对照表

### 3.1 训练入口与配置键

| 现状（vendored verl） | Agent-R1（verl 0.7.0） |
|---|---|
| `python3 -m verl.trainer.main config=examples/config.yaml` | `python3 -m agent_r1.trainer.main_agent_ppo`（Hydra overrides，从 Agent-R1 仓库根目录运行） |
| `worker.actor.model.model_path` | `actor_rollout_ref.model.path` |
| `worker.actor.model.enable_gradient_checkpointing` | `actor_rollout_ref.model.enable_gradient_checkpointing` |
| `worker.reward.reward_function=path:name` | `custom_reward_function.path` + `custom_reward_function.name`（+ 可选 `.reward_kwargs`） |
| `worker.rollout.n` | `actor_rollout_ref.rollout.n` |
| `data.rollout_batch_size` | `data.train_batch_size`（每步 unique prompt 数） |
| `worker.actor.global_batch_size` | `actor_rollout_ref.actor.ppo_mini_batch_size` |
| `worker.actor.micro_batch_size_per_device_for_update` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` |
| `worker.actor.micro_batch_size_per_device_for_experience` | `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` |
| `worker.rollout.tensor_parallel_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| `worker.rollout.gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| `algorithm.use_kl_loss` / `kl_penalty` / `kl_coef` | `actor_rollout_ref.actor.use_kl_loss=True` / 默认 `low_var_kl` / `algorithm.kl_ctrl.kl_coef` |
| `algorithm.disable_kl` | （GRPO 下用 `actor_rollout_ref.actor.use_kl_loss=False` 或 `algorithm.use_kl_in_reward=False`） |
| `trainer.save_checkpoint_path` | `trainer.default_local_dir` |
| `trainer.load_checkpoint_path` | `trainer.resume_mode=resume_path` + `trainer.resume_from_path` |
| `trainer.save_limit` | `trainer.max_actor_ckpt_to_keep` |
| `trainer.val_freq=-1` | `trainer.test_freq=-1` |
| —（GRPO 无需） | 必须显式 `critic.enable=False`、`reward_model.enable=False` |
| — | `data.return_raw_chat=True`（prompt 为 chat 列表） |
| — | `algorithm.norm_adv_by_std_in_grpo=True`（原始 GRPO 归一化） |
| `data.prompt_key` / `data.answer_key` | **不存在**。列名固定：`prompt`、`data_source`、`reward_model.ground_truth`、`extra_info` |

### 3.2 Reward 接口差异

```python
# 现状（批量式，reward/verl_reward_wrapper.py）
def compute_score(reward_inputs: List[Dict[str, str]], **kwargs) -> List[Dict[str, float]]:
    # 每条 reward_input: {"response": 生成的攻击文本, "ground_truth": sensitive, strategy/task_type/...}

# Agent-R1（单样本式）
def compute_score(data_source: str, solution_str: str, ground_truth: Any,
                  extra_info: dict | None, **kwargs) -> float:
    # solution_str == 原接口的 "response"（模型生成的攻击文本）
    # ground_truth == 原接口的 "ground_truth"（= 数据行的 reward_model.ground_truth）
    # extra_info  == 数据行写入 extra_info 列的所有元数据（strategy/task_type/agent_type/...）
```

### 3.3 数据集差异

```jsonc
// 现状（JSON）：字段名自由，answer_key 指定
{"prompt": "…", "sensitive": "…", "strategy": "direct", "task_type": "file_edit", "agent_type": "…"}

// Agent-R1（parquet，固定列）：
{
  "data_source": "privacy_attack",
  "prompt": [{"role": "user", "content": "…"}],          // chat messages 列表（return_raw_chat=True）
  "reward_model": {"style": "rule", "ground_truth": "…"}, // 原 sensitive 字段放这里
  "extra_info": {"strategy": "direct", "task_type": "file_edit", "agent_type": "…"}  // 其余元数据全部放这里
}
```

## 4. 迁移步骤

### 4.1 环境准备

使用 conda **AgentPrivacy** 环境（已确认：Python 3.10.20、vllm 0.11.0、ray 2.55.1 —— 均满足 verl 0.7.0 要求：Python≥3.10、vllm 0.8.5~0.12.0、ray≥2.41）。**只需补装 verl 包本身，无需其他依赖**：

```bash
conda activate AgentPrivacy

# 1. 安装 verl==0.7.0（AgentPrivacy 中缺失，底层依赖已就绪）
pip install "verl==0.7.0"
python3 -c "import verl; print(verl.__file__)"   # 确认 import 到 site-packages，而不是仓库内 vendored verl

# 2. Agent-R1 直接运行，无需 pip install agent_r1
cd /home/fangzibang/Agent-R1 && python3 -c "import agent_r1; print('ok')"
```

**⚠️ 重要**：仓库根目录的 `verl/`（vendored）会因 cwd 优先导入而 shadow pip 安装的 verl 0.7.0。所以训练命令必须**从 Agent-R1 仓库根目录启动**（见 4.4），或者在迁移完成后把本仓库的 `verl/` 目录改名/删除（`git rm -r verl/` 或 `mv verl verl_vendored_bak`），避免误导入旧代码。

### 4.2 数据格式转换（JSON → parquet）

新增 `scripts/convert_json_to_parquet.py`（放在 `build_dataset.py` 之后、训练之前调用）：

```python
"""Convert SelfEvolvingPrivacyRL JSON datasets to Agent-R1 (verl 0.7.0) parquet."""
import json
import sys

import pandas as pd

def convert(json_path: str, parquet_path: str, data_source: str = "privacy_attack") -> None:
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    rows = []
    for r in records:
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "reward_model": {"style": "rule", "ground_truth": r.get("sensitive", "")},
            # 其余字段全部进 extra_info，供 reward 函数读取
            "extra_info": {k: v for k, v in r.items() if k not in ("prompt", "sensitive")},
        })
    pd.DataFrame(rows).to_parquet(parquet_path)
    print(f"[convert] {json_path} -> {parquet_path}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
```

要点：

- **`ground_truth` 语义完全保留**：standard 模式是明文 sensitive 字符串；PrivacyLens / AgentDojo 模式的 JSON 元数据字符串（`{"agent_type": "privacylens", ...}`）也原样放进 `reward_model.ground_truth`，reward 函数里的 `_parse_pl_meta` / `_parse_agentdojo_meta` 无需改动。
- `extra_info` 必须 JSON 可序列化（dict/list/str/int 均可）；不要放 numpy 对象。
- 训练/验证用两套文件：`data.train_files` / `data.val_files` 分别指向 `train.parquet` / `val.parquet`。

### 4.3 Reward 函数改写（新增 adapter，复用全部现有逻辑）

新增 `reward/agentr1_reward_wrapper.py`（**约 25 行**，其余 560 行 defender 逻辑零改动）：

```python
"""Agent-R1 (verl 0.7.0) 单样本 reward 入口，适配现有批量式 wrapper。

verl 0.7.0 通过 importlib 按文件路径加载本模块（custom_reward_function.path），
模块目录不一定在 sys.path 上，故显式把项目根目录加入 sys.path。
"""
import os
import sys
from typing import Any, Dict, List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict | None = None,
    **kwargs,
) -> float:
    """单样本入口：组回原批量接口的 reward_input dict，委托给现有实现。

    - solution_str: 攻击者模型生成的攻击文本（原接口的 "response"）
    - ground_truth: 数据行 reward_model.ground_truth（原接口的 "ground_truth"）
    - extra_info:   数据行 extra_info 列（原接口的顶层元数据字段）
    """
    from reward.verl_reward_wrapper import compute_score as _batch_compute

    reward_input: Dict[str, Any] = {"response": solution_str, "ground_truth": ground_truth}
    if extra_info:
        reward_input.update(extra_info)  # strategy / task_type / agent_type / user_instruction ...

    results: List[Dict[str, float]] = _batch_compute([reward_input])
    return float(results[0].get("overall", 0.0))
```

说明：

- 旧 `compute_score` 内部本来就是逐样本处理（按 mode 分流：standard / privacylens / agentdam / agentdojo / agentdojo_simple），所以单条调用完全安全；guard vLLM、PrivacyLens sandbox、AgentDAM 的 HTTP 调用和所有 `GUARD_*` / `PL_*` 环境变量照旧。
- GRPO 使用的奖励值即返回值 `overall`。如想额外记录 `leaked` / `task_completed` 到日志，可返回 dict（含 `reward_score` 键），具体以 verl 0.7.0 `NaiveRewardManager.run_single` 的约定为准；稳妥起见主路径返回 float。
- 配置中 `custom_reward_function.path=$PROJECT_DIR/reward/agentr1_reward_wrapper.py`、`custom_reward_function.name=compute_score`。

### 4.4 训练启动命令（参数映射示例）

```bash
cd /home/fangzibang/Agent-R1   # 必须从 Agent-R1 根目录运行（避免 vendored verl shadow）

python3 -m agent_r1.trainer.main_agent_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files=/home/fangzibang/SelfEvolvingPrivacyRL/data/train.parquet \
  data.val_files=/home/fangzibang/SelfEvolvingPrivacyRL/data/val.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.enable=False \
  reward_model.enable=False \
  custom_reward_function.path=/home/fangzibang/SelfEvolvingPrivacyRL/reward/agentr1_reward_wrapper.py \
  custom_reward_function.name=compute_score \
  trainer.project_name=SelfEvolvingPrivacyRL \
  trainer.experiment_name=privacy_grpo \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=50 \
  trainer.test_freq=-1 \
  trainer.save_freq=10 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  '+actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
```

> **⚠️ `total_epochs` 语义（verl 0.7.0 与旧版不同）**：训练循环是 `for epoch in range(current_epoch, total_epochs)`，步数上限 = `total_epochs × len(dataloader)`；`total_training_steps` 只负责在 `global_steps >= total_training_steps` 时触发「最后一步」并 `return`。所以**必须把 `total_epochs` 设足够大（如 1000），用 `total_training_steps` 控制实际步数**；设 `total_epochs=1` 会导致每轮只能训 1 步，resume 后甚至直接退出（迭代脚本已按此修正）。

对应关系（与原 `examples/config.yaml` + `iterative_train_with_verl.sh` 的 TRAIN_ARGS）：

| 原值 | 新值 | 备注 |
|---|---|---|
| `data.rollout_batch_size=32` | `data.train_batch_size=32` | 每步 unique prompt 数 |
| `worker.rollout.n=4` | `actor_rollout_ref.rollout.n=4` | GRPO group size |
| `worker.actor.global_batch_size=32` | `actor_rollout_ref.actor.ppo_mini_batch_size=32` | 每 mini-batch 序列数 |
| `worker.actor.micro_batch_size_per_device_for_update=1` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1` | |
| `worker.actor.micro_batch_size_per_device_for_experience=1` | `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1` | |
| `worker.rollout.tensor_parallel_size=2` | `actor_rollout_ref.rollout.tensor_model_parallel_size=2` | |
| `worker.rollout.gpu_memory_utilization=0.4` | `actor_rollout_ref.rollout.gpu_memory_utilization=0.4` | |
| `worker.actor.model.model_path=$BASE_MODEL` | `actor_rollout_ref.model.path=$BASE_MODEL` | |
| `worker.reward.reward_function=...verl_reward_wrapper.py:compute_score` | `custom_reward_function.path/name` | 指向新 adapter |
| `algorithm.use_kl_loss=true, kl_coef=1e-2` | `actor_rollout_ref.actor.use_kl_loss=True`（kl_coef 默认 1e-2） | |
| `trainer.save_checkpoint_path=$SAVE_PATH` | `trainer.default_local_dir=$SAVE_PATH` | |
| `trainer.val_freq=-1` | `trainer.test_freq=-1` | |
| `trainer.max_steps` | `trainer.total_training_steps` + `trainer.total_epochs=1000` | **语义不同**：total_epochs 是循环上限（设大），total_training_steps 触发提前结束（见下） |
| — | `critic.enable=False`, `reward_model.enable=False` | GRPO 必须，否则报错 |
| — | `data.return_raw_chat=True` | prompt 是 chat 列表时必须 |
| — | `algorithm.norm_adv_by_std_in_grpo=True` | 复现原 GRPO 行为 |
| — | `actor_rollout_ref.rollout.agent.num_workers=2` | 默认 8；无工具环境可调小 |

### 4.5 迭代训练脚本改写

把 `training/iterative_train_with_verl.sh` 的三处改动点（其余流程：数据生成、round 循环、`LOAD_CKPT` 查找逻辑均保留）：

1. **训练命令**：`python3 -m verl.trainer.main "${TRAIN_ARGS[@]}"` → `(cd "$AGENT_R1_ROOT" && python3 -m agent_r1.trainer.main_agent_ppo "${TRAIN_ARGS[@]}")`。
2. **数据**：`build_dataset.py` 之后插入 JSON→parquet 转换（4.2 的脚本）。
3. **checkpoint 加载**：

```bash
if [[ -n "$LOAD_CKPT" ]]; then
  TRAIN_ARGS+=(trainer.resume_mode=resume_path)
  TRAIN_ARGS+=(trainer.resume_from_path=$LOAD_CKPT)   # 路径含 global_step_，与原查找逻辑兼容
fi
```

4. **保存 HF 权重供下轮数据生成**：verl 0.7.0 checkpoint 默认只存 sharded `model/optimizer/extra`（`global_step_N/actor/model_world_size_*_rank_*.pt`），**下一轮 `build_dataset.py` 用 vLLM 直接加载 ckpt 目录会失败**。必须追加：

```bash
TRAIN_ARGS+=(actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']")
```

这样每轮保存时 rank 0 会同时写 `global_step_N/actor/huggingface/`（HF 全量），`MODEL_FOR_DATA` 可直接指向该目录（如不够，指向 `.../actor/huggingface/`）。

### 4.6 冒烟验证

先小规模跑通链路再正式训练：

```bash
# 1) 最小数据集 + mock guard（不调 LLM，快速验证 reward 链路）
GUARD_USE_MOCK=true \
  bash -c 'ROUNDS=1 SAMPLES_PER_ROUND=8 STEPS_PER_ROUND=5 \
  bash /home/fangzibang/SelfEvolvingPrivacyRL/training/iterative_train_with_agentr1.sh'

# 2) 验证要点
#    - reward 打印：每条 sample 出现 attack/response/secret/overall
#    - checkpoint：$SAVE_PATH/global_step_*/actor/ 存在且含 huggingface/
#    - 第二轮 resume：日志显示从 global_step_N 恢复
# 3) 然后关掉 mock，用小批量 + 真实 guard vLLM 跑 1 轮
```

## 5. 注意事项与坑

1. **vendored `verl/` shadow 问题（最容易踩）**：训练必须从 `/home/fangzibang/Agent-R1` 目录启动；从本仓库目录启动 `-m verl.*` / `-m agent_r1.*` 会导入仓库内旧 verl（甚至 `import verl` 会因缺 `codetiming` 直接报错）。迁移完成后建议 `mv verl verl_vendored_bak` 或 `git rm -r verl`。
2. **`data.max_prompt_length` 要够**：attack meta-prompt 较长，保持 512 或放宽到 1024，并开 `data.filter_overlong_prompts=True`（超长会被过滤而非崩溃）。
3. **reward 函数在 ray worker 里运行**：`GUARD_OPENAI_URL`、`GUARD_MODEL`、`GUARD_TIMEOUT`、`PL_SANDBOX_USE_MOCK`、`PRIVACYLENS_HOME`、`PROCODER_HOME` 等环境变量必须随训练进程导出（`scripts/common.sh` 已做，沿用即可）；新增 wrapper 里显式 `sys.path.insert` 项目根目录，保证 `from reward.interface import RewardInterface` 等导入可用。
4. **GRPO 必填项**：`critic.enable=False`、`reward_model.enable=False`、`algorithm.adv_estimator=grpo`、`actor_rollout_ref.rollout.n>=2`，否则报错或没有组内相对 advantage。
5. **守护进程清理**：`scripts/common.sh` 的 `cleanup_stale_gpu_processes` 目前 `pkill -f "verl.trainer.main"`，需加一行 `pkill -9 -f "agent_r1.trainer.main_agent_ppo"`，否则残留 ray 进程会占 GPU。
6. **单步流程无需 AgentFlow**：本项目是「模型生成攻击文本 → 外部 defender（HTTP/沙箱）判奖励」，没有多轮工具交互，用默认 `single_step_agent` 即可，不需要写 `recipes/.../base.yaml` 或注册 AgentFlow。（若未来要做「defender 轨迹级训练」的多轮方案，再参考 `recipes/hotpotqa/` 的 AgentFlow 写法。）
7. **旧配置文件废弃**：`examples/config.yaml`、`examples/config_privacypeek*.yaml` 的 `worker.*` 键在 verl 0.7.0 下不存在，迁移后不再使用（可保留作历史记录）。
8. **defender（guard）训练同样适用**：`scripts/guard_train.sh` 走同一迭代脚本 + `REWARD_PERSPECTIVE=defender`，迁移方式完全一致（reward 函数体不变，只是视角/权重不同）。
9. **GPU 布局：40GB 单卡放不下 FSDP 4B + vLLM**。实测单卡（A100-40GB）时 FSDP 单卡全量权重+梯度占 ~31GB，vLLM 无法启动；**必须用 2 卡**（`ATTACKER_GPU=0,1` → `n_gpus_per_node=2`，FSDP 双卡 shard 每卡 ~8.6GB，vLLM tp=1 + `gpu_memory_utilization=0.5` 可行）。guard vLLM 服务仍在独立卡（默认 GPU 2 / 端口 5000）。
10. **环境已就绪，只缺 verl 包**：`AgentPrivacy` 里 vllm 0.11.0 / ray 2.55.1 均满足 verl 0.7.0 的版本要求，补装 `pip install verl==0.7.0` 即可；安装时如提示依赖冲突，以 verl 0.7.0 要求的 vllm/ray 版本范围为准。
11. **⚠️ 必须 patch verl 0.7.0 的 vLLM max_model_len**（已应用，重装 verl 后需重新打）：`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 第 198 行无条件用 HF `max_position_embeddings` 覆盖用户设置——Qwen3 是 **262144**，40GB 卡上需要 36GB KV cache 直接启动失败。patch 后仅在 `config.max_model_len is None` 时用 HF 默认值；训练时显式传 `actor_rollout_ref.rollout.max_model_len=2048`（≥ max_prompt_length+max_response_length）。
12. **logger 必须去掉 wandb**：默认 `trainer.logger=["console","wandb"]`，未登录 wandb 会 `UsageError`。传 `trainer.logger='["console"]'`（或登录 wandb）。
13. **磁盘**：checkpoint 含 `hf_model` 后单步 ~63GB（model+optimizer shards 各 ~20GB + HF 全量 8GB ×2）。冒烟测试和 ray session 别放根分区（实测根分区被 ray spill 写满导致 ENOSPC 中断训练）；迭代脚本的 `SAVE_PATH` 在 `/home/fangzibang/data_32T`（独立 32T 盘）不受影响。训练完成后及时清理测试 checkpoint。
14. **dataclass 注入的配置键需要 `+` 前缀**：`actor_rollout_ref.actor.checkpoint.save_contents`（CheckpointConfig dataclass 提供，不在 yaml 中）必须写成 `'+actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'`，否则 Hydra 报 "Key is not in struct"。

## 6. 改动文件清单

| 文件 | 动作 |
|---|---|
| `reward/agentr1_reward_wrapper.py` | **新增**：单样本 adapter（4.3 的代码） |
| `scripts/convert_json_to_parquet.py` | **新增**：JSON→parquet 转换（4.2 的代码） |
| `training/iterative_train_with_agentr1.sh` | **新增**（或原地改 `iterative_train_with_verl.sh`）：Agent-R1 版迭代训练（4.4/4.5） |
| `scripts/common.sh` | 修改：cleanup 加 `pkill -f agent_r1.trainer.main_agent_ppo` |
| `~/.conda/envs/AgentPrivacy/.../verl/workers/rollout/vllm_rollout/vllm_async_server.py` | **已 patch**（max_model_len 覆盖逻辑，见注意事项 11；重装 verl 后需重新打） |
| `verl/`（vendored） | 建议改名/删除，防止 shadow verl 0.7.0 |
| `examples/config.yaml` / `config_privacypeek*.yaml` | 废弃（保留作记录） |

### 冒烟验证结果（2026-08-16 已实测通过）

在 `AgentPrivacy` 环境 + GPU 0,1（A100-40GB×2）上，用 8 条样本 + mock guard 跑通：

```
step:1 - actor/entropy:0.614 - critic/score/mean:0.0825 - response_length/mean:86.4
        （rollout → RewardLoopWorker 自定义 reward → GRPO update → save checkpoint）
step:2 - training/global_step:2 - training/epoch:2  （resume_mode=resume_path 从 global_step_1 恢复）
'Final validation metrics: None'   （is_last_step 正常收尾，EXIT=0）
```

- ✅ vLLM rollout（patch 后 max_model_len=2048）
- ✅ 自定义 reward（AgentDojo Simple / standard 模式分流，`critic/score/mean` 正常上报）
- ✅ checkpoint 完整保存：`model/optimizer/extra_state` shards + `actor/huggingface/`（HF 全量，供下轮 `build_dataset.py` 直接加载）
- ✅ `latest_checkpointed_iteration.txt` 写入（`resume_mode=auto` 也兼容）

## 参考

- Agent-R1 仓库：`/home/fangzibang/Agent-R1`（`examples/hotpotqa/run_grpo.sh`、`examples/gsm8k/run_steppo.sh`、`recipes/gsm8k/reward_fn.py`、`docs/`）
- verl 0.7.0 checkpoint 配置：[verl checkpoint 文档](https://github.com/verl-project/verl/blob/59f53cc4/docs/advance/checkpoint.rst)（`resume_mode` / `resume_from_path` / `save_contents`），[resume 相关 issue](https://github.com/verl-project/verl/issues/4534)
- verl 安装指南：<https://verl.readthedocs.io/en/latest/start/install.html>
