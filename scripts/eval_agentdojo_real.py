#!/usr/bin/env python3
"""Evaluate AgentDojo defender with a real vLLM guard model.

Usage:
    PYTHONPATH=/home/fangzibang/agentdojo/src:$PYTHONPATH \\
    GUARD_PORT=5000 GUARD_MODEL=guard-llama3b \\
    python scripts/eval_agentdojo_real.py \\
        --suite workspace --limit 20 --max-steps 6

Results are saved to runs/agentdojo_eval_*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AgentDojo defender")
    parser.add_argument("--suite", type=str, default="workspace",
                        help="AgentDojo suite name")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max number of cases to run")
    parser.add_argument("--max-steps", type=int, default=6,
                        help="Max ReAct steps per case")
    parser.add_argument("--user-tasks", type=str, default=None,
                        help="Comma-separated user task IDs (default: all)")
    parser.add_argument("--injection-tasks", type=str, default=None,
                        help="Comma-separated injection task IDs (default: all)")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Output directory for results")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, "/home/fangzibang/agentdojo/src")

    os.environ.setdefault("GUARD_PORT", "5000")
    os.environ.setdefault("GUARD_MODEL", "guard-llama3b")

    guard_port = os.environ["GUARD_PORT"]
    guard_model = os.environ["GUARD_MODEL"]

    print(f"Guard model: {guard_model} on port {guard_port}")
    print(f"Suite: {args.suite}, limit: {args.limit}, max_steps: {args.max_steps}")

    from defender.agentdojo_env import run_agentdojo_agent
    from reward.agentdojo_reward import get_agentdojo_reward

    # ------------------------------------------------------------------
    # Load tasks
    # ------------------------------------------------------------------
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", args.suite)
    user_tasks = suite.user_tasks
    injection_tasks = suite.injection_tasks

    print(f"Loaded suite '{args.suite}': "
          f"{len(user_tasks)} user tasks, {len(injection_tasks)} injection tasks")

    # Filter tasks
    if args.user_tasks:
        ut_filter = {t.strip() for t in args.user_tasks.split(",")}
        user_tasks = {k: v for k, v in user_tasks.items() if k in ut_filter}
    if args.injection_tasks:
        it_filter = {t.strip() for t in args.injection_tasks.split(",")}
        injection_tasks = {k: v for k, v in injection_tasks.items() if k in it_filter}
    else:
        # Default: only first 5 injection tasks for speed
        it_keys = sorted(injection_tasks.keys())[:5]
        injection_tasks = {k: injection_tasks[k] for k in it_keys}

    # Build case list
    cases: List[Dict[str, Any]] = []
    for ut_id in sorted(user_tasks.keys()):
        for inj_id in sorted(injection_tasks.keys()):
            cases.append({
                "suite_name": args.suite,
                "user_task_id": ut_id,
                "injection_task_id": inj_id,
                "case_id": f"{args.suite}_{ut_id}_{inj_id}",
            })

    # Limit
    if args.limit and len(cases) > args.limit:
        import random
        random.Random(42).shuffle(cases)
        cases = cases[:args.limit]

    print(f"Evaluating {len(cases)} cases...")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    scorer = get_agentdojo_reward()
    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, case in enumerate(cases):
        case_start = time.time()
        print(f"\n[{i+1}/{len(cases)}] {case['case_id']}")

        agent_result = run_agentdojo_agent(
            suite_name=case["suite_name"],
            user_task_id=case["user_task_id"],
            injection_task_id=case["injection_task_id"],
            injections={},
            first_action=None,  # Let guard LLM generate all actions
            max_steps=args.max_steps,
        )

        reward = scorer.compute_reward(agent_result)

        row = {
            **case,
            "utility": agent_result["utility"],
            "security": agent_result["security"],
            "n_tool_calls": agent_result.get("n_tool_calls", 0),
            "model_output": agent_result.get("model_output", "")[:500],
            "reward": reward,
            "duration_s": round(time.time() - case_start, 1),
            "error": agent_result.get("error"),
        }
        results.append(row)

        util_icon = "OK" if row["utility"] else "!!"
        sec_icon = "SAFE" if row["security"] else "FAIL"
        print(f"  utility={util_icon}  security={sec_icon}  "
              f"reward={reward['overall']:.3f}  tools={row['n_tool_calls']}  "
              f"{row['duration_s']}s")
        if row["error"]:
            print(f"  ERROR: {row['error'][:200]}")

    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(results)
    n_utility = sum(1 for r in results if r["utility"])
    n_security = sum(1 for r in results if r["security"])
    n_both = sum(1 for r in results if r["utility"] and r["security"])
    n_neither = sum(1 for r in results if not r["utility"] and not r["security"])
    avg_reward = sum(r["reward"]["overall"] for r in results) / max(n, 1)
    avg_tools = sum(r["n_tool_calls"] for r in results) / max(n, 1)
    errors = [r for r in results if r["error"]]

    print("\n" + "=" * 60)
    print(f"Results for {args.suite} ({n} cases, {total_time:.0f}s)")
    print(f"  Utility rate:  {n_utility}/{n} = {100*n_utility/n:.1f}%")
    print(f"  Security rate: {n_security}/{n} = {100*n_security/n:.1f}%")
    print(f"  Both (ideal):  {n_both}/{n} = {100*n_both/n:.1f}%")
    print(f"  Neither:       {n_neither}/{n} = {100*n_neither/n:.1f}%")
    print(f"  Avg reward:    {avg_reward:.3f}")
    print(f"  Avg tools:     {avg_tools:.1f}")
    print(f"  Errors:        {len(errors)}/{n}")
    print(f"  Avg latency:   {total_time/max(n,1):.1f}s per case")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"agentdojo_eval_{args.suite}_{ts}.json"
    output_path.write_text(json.dumps({
        "config": {
            "suite": args.suite,
            "guard_model": guard_model,
            "guard_port": guard_port,
            "max_steps": args.max_steps,
            "limit": args.limit,
        },
        "summary": {
            "n": n,
            "utility_rate": n_utility / max(n, 1),
            "security_rate": n_security / max(n, 1),
            "both_rate": n_both / max(n, 1),
            "neither_rate": n_neither / max(n, 1),
            "avg_reward": avg_reward,
            "avg_tools": avg_tools,
            "errors": len(errors),
            "total_time_s": total_time,
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
