"""Agent-R1 (verl 0.7.0) 单样本 reward 入口，适配现有批量式 wrapper。

verl 0.7.0 通过 importlib 按文件路径加载本模块（custom_reward_function.path），
模块所在目录不保证在 sys.path 上，故显式把项目根目录加入 sys.path。
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
) -> Dict[str, float]:
    """单样本入口：组回原批量接口的 reward_input dict，委托给现有实现。

    - solution_str: 攻击者模型生成的攻击文本（原接口的 "response"）
    - ground_truth: 数据行 reward_model.ground_truth（原接口的 "ground_truth"）
    - extra_info:   数据行 extra_info 列（原接口的顶层元数据字段）

    返回 dict（verl 0.7.0 约定：``score`` 键为 GRPO 主奖励，其余键进入
    reward_extra_info，被 trainer 聚合成每步 ``reward/*`` 指标，与旧版 verl
    的 experiment_log.jsonl 输出保持一致）。
    """
    from reward.verl_reward_wrapper import compute_score as _batch_compute

    reward_input: Dict[str, Any] = {"response": solution_str, "ground_truth": ground_truth}
    if extra_info:
        reward_input.update(extra_info)  # strategy / task_type / agent_type / user_instruction ...

    results: List[Dict[str, float]] = _batch_compute([reward_input])
    result = results[0] if results else {}

    overall = float(result.get("overall", 0.0))
    leaked = float(result.get("leaked", 0.0))
    task_completed = float(result.get("task_completed", 0.0))
    # 非 agentdojo_simple 模式没有这两个维度，用 leaked 近似（保持 5 维度聚合）
    attack_success = float(result.get("attack_success", leaked))
    privacy_leak = float(result.get("privacy_leak", leaked))
    injection_visible = float(result.get("injection_visible", 0.0))
    environment_error = float(result.get("environment_error", 0.0))
    attack_prefix_completion = float(result.get("attack_prefix_completion", 0.0))
    stage1_success = float(result.get("stage1_success", 0.0))
    invalid = float(result.get("invalid", 0.0))
    return {
        "score": overall,
        "overall": overall,
        "leaked": leaked,
        "task_completed": task_completed,
        "attack_success": attack_success,
        "privacy_leak": privacy_leak,
        "injection_visible": injection_visible,
        "environment_error": environment_error,
        "attack_prefix_completion": attack_prefix_completion,
        "stage1_success": stage1_success,
        "invalid": invalid,
    }
