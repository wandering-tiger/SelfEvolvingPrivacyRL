#!/usr/bin/env python
"""End-to-end test for the PrivacyLens ToolEmu sandbox integration.

Covers the complete pipeline:
  1. Scenario generation (attacker side)
  2. Mock sandbox execution (no vLLM needed)
  3. Real sandbox execution (requires vLLM server)
  4. Reward computation with static leakage detection
  5. Dataset record generation for training

Usage::

    # Mock mode (no vLLM needed — fast smoke test)
    conda run -n AgentPrivacy python tests/test_privacylens_sandbox.py --mock

    # Real mode (requires vLLM server running)
    conda run -n AgentPrivacy python tests/test_privacylens_sandbox.py \\
        --port 5000 --model_name Qwen/Qwen3-32B

    # With auto-started vLLM server
    conda run -n AgentPrivacy python tests/test_privacylens_sandbox.py \\
        --model_path /path/to/model --port 5000 --gpu_id 0

Environment Variables
---------------------
GUARD_OPENAI_URL : str
    vLLM chat-completions endpoint. Overrides --port.
GUARD_MODEL : str
    Model name sent in the payload.
GUARD_USE_MOCK : str
    Set to "true" to bypass all LLM calls.
PL_SANDBOX_USE_MOCK : str
    Set to "true" to use mock trajectory (faster than real sandbox).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ============================================================================
# vLLM server helpers
# ============================================================================

def _wait_for_server(base_url: str, timeout: int = 180,
                     proc=None, log_file=None) -> None:
    """Poll the vLLM endpoint until it responds or times out."""
    import requests
    deadline = time.time() + timeout
    payload = {
        "model": "dummy", "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 1,
    }
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            msg = f"vLLM exited with code {proc.returncode}"
            if log_file and os.path.exists(log_file):
                with open(log_file) as f:
                    msg += f"\nLog:\n{f.read()[-2000:]}"
            raise RuntimeError(msg)
        try:
            r = requests.post(base_url, json=payload, timeout=5)
            if r.status_code == 200:
                print(f"  vLLM ready at {base_url}")
                return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM timeout after {timeout}s: {base_url}")


def _start_vllm_server(model_path: str, port: int, gpu_id: int,
                       gpu_mem_util: float = 0.8,
                       max_model_len: int = 8192):
    """Start a vLLM server as a subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = os.path.join(REPO_ROOT, f"vllm_pl_test_port{port}.log")
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "vllm_service_init/start_vllm_server.py"),
        "--port", str(port),
        "--model_path", model_path,
        "--gpu_mem_util", str(gpu_mem_util),
        "--max_model_len", str(max_model_len),
    ]
    f = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f, start_new_session=True)
    return proc, log_file


# ============================================================================
# Test helpers
# ============================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ FAIL: {msg}")


# ============================================================================
# Tests
# ============================================================================

class PrivacyLensSandboxTester:
    """Runs all PrivacyLens integration tests."""

    def __init__(self, mock: bool = False, verbose: bool = False):
        self.mock = mock
        self.verbose = verbose
        self.passed = 0
        self.failed = 0

    def check(self, condition: bool, msg: str) -> bool:
        if condition:
            self.passed += 1
            _ok(msg)
        else:
            self.failed += 1
            _fail(msg)
        return condition

    # ------------------------------------------------------------------
    # Test 1: Scenario generation
    # ------------------------------------------------------------------
    def test_scenario_generation(self) -> None:
        _banner("Test 1: Scenario Generation")
        from attacker.scenario_generator import ScenarioGenerator

        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate()

        self.check(
            "privacy_sensitive_context" in scenario,
            "scenario has privacy_sensitive_context",
        )
        self.check(
            "services_data" not in scenario,
            "scenario has NO deprecated services_data",
        )
        self.check(
            len(scenario["sensitive_info_items"]) >= 1,
            f"scenario has {len(scenario['sensitive_info_items'])} sensitive items",
        )
        self.check(
            len(scenario["toolkits"]) >= 1,
            f"scenario has toolkits: {scenario['toolkits']}",
        )
        self.check(
            "Context:" in scenario["privacy_sensitive_context"],
            "privacy_sensitive_context starts with 'Context:'",
        )
        self.check(
            "Detailed Plot:" in scenario["privacy_sensitive_context"],
            "privacy_sensitive_context contains 'Detailed Plot:'",
        )
        self.check(
            len(scenario["privacy_sensitive_context"]) > 200,
            f"privacy_sensitive_context is detailed ({len(scenario['privacy_sensitive_context'])} chars)",
        )

        # Batch generation
        batch = gen.generate_batch(5)
        self.check(
            len(batch) == 5,
            f"generate_batch(5) returns {len(batch)} scenarios",
        )
        for s in batch:
            self.check(
                "privacy_sensitive_context" in s and "sensitive_info_items" in s,
                f"batch scenario {s['final_action_type']} has required fields",
            )

        if self.verbose:
            s = scenario
            print(f"\n  Sample scenario:")
            print(f"    user_name:      {s['user_name']}")
            print(f"    user_email:     {s['user_email']}")
            print(f"    toolkits:       {s['toolkits']}")
            print(f"    final_action:   {s['final_action_type']}")
            print(f"    user_instruction: {s['user_instruction']}")
            print(f"    sensitive_items:  {s['sensitive_info_items']}")
            print(f"    context (first 120 chars): {s['privacy_sensitive_context'][:120]}...")

    # ------------------------------------------------------------------
    # Test 2: Dataset record generation
    # ------------------------------------------------------------------
    def test_dataset_records(self) -> None:
        _banner("Test 2: Dataset Record Generation")
        from attacker.dataset_builder import build_pl_style_records

        records = build_pl_style_records(num_samples=5, attack_strategies=["direct", "indirect"])
        self.check(len(records) == 5, f"build_pl_style_records(5) returns {len(records)} records")

        for i, rec in enumerate(records):
            self.check(
                rec.get("agent_type") == "privacylens",
                f"record {i}: agent_type='privacylens'",
            )
            self.check(
                "privacy_sensitive_context" in rec,
                f"record {i}: has privacy_sensitive_context",
            )
            self.check(
                "services_data" not in rec,
                f"record {i}: no deprecated services_data",
            )
            self.check(
                len(rec.get("sensitive_info_items", [])) >= 1,
                f"record {i}: has {len(rec.get('sensitive_info_items', []))} sensitive items",
            )
            self.check(
                len(rec.get("prompt", "")) > 0,
                f"record {i}: has attacker prompt ({len(rec['prompt'])} chars)",
            )
            self.check(
                len(rec.get("toolkits", [])) >= 1,
                f"record {i}: has toolkits {rec['toolkits']}",
            )

    # ------------------------------------------------------------------
    # Test 3: Mock sandbox
    # ------------------------------------------------------------------
    def test_mock_sandbox(self) -> None:
        _banner("Test 3: Mock Sandbox Execution")
        from attacker.scenario_generator import ScenarioGenerator
        from defender.agent import run_pl_agent

        os.environ["PL_SANDBOX_USE_MOCK"] = "true"
        os.environ["GUARD_USE_MOCK"] = "true"

        gen = ScenarioGenerator(seed=123)
        scenario = gen.generate()

        result = run_pl_agent(
            user_instruction=scenario["user_instruction"],
            user_name=scenario["user_name"],
            user_email=scenario["user_email"],
            privacy_sensitive_context=scenario["privacy_sensitive_context"],
            toolkits=scenario["toolkits"],
            final_action_type=scenario["final_action_type"],
            sensitive_info_items=scenario["sensitive_info_items"],
            max_steps=10,
            return_metadata=True,
            verbose=self.verbose,
        )

        self.check(
            isinstance(result, dict),
            "run_pl_agent returns dict (return_metadata=True)",
        )
        self.check(
            "final" in result and len(result["final"]) > 0,
            f"has final action ({len(result.get('final', ''))} chars)",
        )
        self.check(
            "pl_format_trajectory" in result,
            "has pl_format_trajectory",
        )
        self.check(
            len(result.get("pl_format_trajectory", "")) > 0,
            f"trajectory has content ({len(result.get('pl_format_trajectory', ''))} chars)",
        )
        self.check(
            "tool_calls" in result,
            f"has tool_calls ({len(result.get('tool_calls', []))} calls)",
        )
        self.check(
            result.get("error") is None or result.get("error") == "",
            f"no error (error={result.get('error', 'none')})",
        )

        if self.verbose:
            print(f"\n  Mock sandbox result:")
            print(f"    final_action: {result.get('final', '')[:200]}")
            print(f"    tool_calls: {len(result.get('tool_calls', []))}")
            print(f"    trajectory:  {result.get('pl_format_trajectory', '')[:300]}")

    # ------------------------------------------------------------------
    # Test 4: Reward computation (static leakage detection)
    # ------------------------------------------------------------------
    def test_static_leakage_detection(self) -> None:
        _banner("Test 4: Static Leakage Detection")
        from reward.interface import RewardInterface

        scorer = RewardInterface()

        # Case A: clear leak (exact substring match)
        sensitive = ["Dave was diagnosed with anxiety"]
        response = "Team update: Dave was diagnosed with anxiety and is doing well."
        score = scorer.detect_leakage(response, sensitive)
        self.check(
            score >= 0.5,
            f"Detected leak: score={score:.2f} (response contains the sensitive phrase)",
        )

        # Case B: no leak
        response_clean = "Team update: Everyone is doing well on the Q2 project."
        score_clean = scorer.detect_leakage(response_clean, sensitive)
        self.check(
            score_clean < 0.5,
            f"No leak: score={score_clean:.2f} (no sensitive data in response)",
        )

        # Case C: partial leak (multi-word sensitive item, some words match)
        sensitive_partial = ["John Smith lives at 742 Evergreen Terrace, Springfield."]
        response_partial = "John Smith is at 742 Evergreen Terrace as you requested."
        score_partial = scorer.detect_leakage(response_partial, sensitive_partial)
        self.check(
            score_partial > 0.0,
            f"Partial leak detected: score={score_partial:.2f}",
        )

        # Case D: empty response
        score_empty = scorer.detect_leakage("", sensitive)
        self.check(
            score_empty == 0.0,
            f"Empty response has no leak: score={score_empty:.2f}",
        )

        # Test compute_reward
        reward = scorer.compute_reward(
            assistant_action=response,
            sensitive_items=sensitive,
            task_completed=True,
        )
        self.check("overall" in reward, f"reward has 'overall': {reward}")
        self.check(reward["leaked"] > 0.0, f"reward.leaked > 0: {reward['leaked']:.2f}")
        self.check(reward["task_completed"] == 1.0, f"task_completed=1.0 when True")

        reward_no_task = scorer.compute_reward(
            assistant_action=response_clean,
            sensitive_items=sensitive,
            task_completed=False,
        )
        self.check(
            reward_no_task["task_completed"] == 0.0,
            f"task_completed=0.0 when False: {reward_no_task}",
        )

        if self.verbose:
            print(f"\n  Leakage detection examples:")
            print(f"    Leak:     {score:.2f}  → '{response[:60]}...'")
            print(f"    No leak:  {score_clean:.2f}  → '{response_clean[:60]}...'")
            print(f"    Partial:  {score_partial:.2f}  → '{response_partial[:60]}...'")

    # ------------------------------------------------------------------
    # Test 5: Real sandbox (if vLLM available)
    # ------------------------------------------------------------------
    def test_real_sandbox(self) -> bool:
        _banner("Test 5: Real Sandbox (vLLM required)")
        from attacker.scenario_generator import ScenarioGenerator

        # Check if vLLM is reachable
        base_url = os.getenv(
            "GUARD_OPENAI_URL", "http://127.0.0.1:8000/v1/chat/completions"
        )
        import requests
        try:
            r = requests.post(
                base_url,
                json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 1},
                timeout=5,
            )
            if r.status_code != 200:
                print("  SKIP: vLLM server not responding with 200")
                return False
        except Exception as e:
            print(f"  SKIP: vLLM server not reachable ({e})")
            return False

        os.environ.pop("PL_SANDBOX_USE_MOCK", None)
        os.environ.pop("GUARD_USE_MOCK", None)

        gen = ScenarioGenerator(seed=456)
        scenario = gen.generate()

        # Use a simpler scenario for faster testing
        scenario["toolkits"] = ["Gmail", "NotionManager"]
        scenario["final_action_type"] = "GmailSendEmail"

        print(f"  Running real sandbox with toolkits={scenario['toolkits']}...")
        print(f"  Instruction: {scenario['user_instruction'][:100]}...")

        t_start = time.time()

        try:
            from defender.privacylens.sandbox import PrivacyLensSandbox, PLSandboxConfig

            config = PLSandboxConfig(
                max_iterations=5,  # short for testing
                agent_type="privacy_enhanced",
                simulator_type="privacy_risk",
                agent_temperature=0.0,
                simulator_temperature=0.0,
                refine_observation=False,
            )
            sandbox = PrivacyLensSandbox(config)
            result = sandbox.run(
                user_name=scenario["user_name"],
                user_email=scenario["user_email"],
                user_instruction=scenario["user_instruction"],
                privacy_sensitive_context=scenario["privacy_sensitive_context"],
                toolkits=scenario["toolkits"],
                sensitive_info_items=scenario["sensitive_info_items"],
                final_action_type=scenario["final_action_type"],
            )
            elapsed = time.time() - t_start
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.check(False, f"Real sandbox crashed: {exc}")
            return False

        self.check(
            result.get("error") is None,
            f"No sandbox error (error={result.get('error', 'none')})",
        )
        self.check(
            len(result.get("trajectory", "")) > 0,
            f"Trajectory has content ({len(result.get('trajectory', ''))} chars)",
        )
        self.check(
            len(result.get("trajectory_steps", [])) >= 1,
            f"At least 1 trajectory step ({len(result.get('trajectory_steps', []))} steps)",
        )
        self.check(
            len(result.get("final_action", "")) > 0,
            f"Has final action ({len(result.get('final_action', ''))} chars)",
        )

        # Check that trajectory contains expected format
        traj = result.get("trajectory", "")
        self.check(
            "Action:" in traj,
            "Trajectory contains 'Action:' blocks",
        )
        self.check(
            "Observation:" in traj,
            "Trajectory contains 'Observation:' blocks",
        )

        print(f"  Real sandbox completed in {elapsed:.1f}s")

        if self.verbose:
            print(f"\n  Real sandbox result:")
            print(f"    Steps:      {len(result.get('trajectory_steps', []))}")
            print(f"    Trajectory: {traj[:500]}")
            print(f"    Final:      {result.get('final_action', '')[:300]}")

        # Test leakage on the real output
        from reward.interface import RewardInterface
        scorer = RewardInterface()
        leak_score = scorer.detect_leakage(
            result["final_action"], scenario["sensitive_info_items"]
        )
        print(f"  Leakage score on real output: {leak_score:.2f}")

        return True

    # ------------------------------------------------------------------
    # Test 6: End-to-end with defender agent interface
    # ------------------------------------------------------------------
    def test_defender_agent_interface(self) -> None:
        _banner("Test 6: Defender Agent Interface (mock)")
        from attacker.scenario_generator import ScenarioGenerator
        from defender.agent import run_pl_agent

        os.environ["PL_SANDBOX_USE_MOCK"] = "true"
        os.environ["GUARD_USE_MOCK"] = "true"

        gen = ScenarioGenerator(seed=789)
        scenarios = gen.generate_batch(3)

        for i, scenario in enumerate(scenarios):
            result = run_pl_agent(
                user_instruction=scenario["user_instruction"],
                user_name=scenario["user_name"],
                user_email=scenario["user_email"],
                privacy_sensitive_context=scenario["privacy_sensitive_context"],
                toolkits=scenario["toolkits"],
                final_action_type=scenario["final_action_type"],
                sensitive_info_items=scenario["sensitive_info_items"],
                max_steps=10,
                return_metadata=True,
            )

            final = result.get("final", "") if isinstance(result, dict) else str(result)
            tools_count = len(result.get("tool_calls", [])) if isinstance(result, dict) else 0
            has_trajectory = len(result.get("pl_format_trajectory", "")) > 0

            self.check(
                isinstance(result, dict) and len(final) > 0,
                f"case {i}: run_pl_agent returns valid result (final={len(final)} chars)",
            )
            self.check(
                has_trajectory,
                f"case {i}: has trajectory content",
            )

            if self.verbose:
                print(f"\n  Case {i}: {scenario['final_action_type']} "
                      f"tools={tools_count} final_len={len(final)} "
                      f"traj_len={len(result.get('pl_format_trajectory', ''))}")

    # ------------------------------------------------------------------
    # Test 7: Import chain verification
    # ------------------------------------------------------------------
    def test_import_chain(self) -> None:
        _banner("Test 7: Import Chain Verification")
        modules_to_check = [
            ("defender.privacylens", "PrivacyLens package (lazy)"),
            ("defender.agent", "Defender agent module"),
            ("attacker.scenario_generator", "Scenario generator"),
            ("attacker.dataset_builder", "Dataset builder"),
            ("attacker.state", "Attack state"),
            ("reward.interface", "Reward interface"),
            ("reward.verl_reward_wrapper", "Reward wrapper"),
        ]
        for mod_name, desc in modules_to_check:
            try:
                __import__(mod_name)
                self.check(True, f"import {desc}")
            except ImportError as e:
                self.check(False, f"import {desc}: {e}")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run_all(self, skip_real: bool = True) -> bool:
        print("PrivacyLens Sandbox Integration Test Suite")
        print(f"  mock={self.mock}  verbose={self.verbose}")

        self.test_import_chain()
        self.test_scenario_generation()
        self.test_dataset_records()
        self.test_mock_sandbox()
        self.test_static_leakage_detection()
        self.test_defender_agent_interface()

        if not skip_real and not self.mock:
            self.test_real_sandbox()

        _banner("Results")
        total = self.passed + self.failed
        print(f"  Passed: {self.passed}/{total}")
        if self.failed > 0:
            print(f"  Failed: {self.failed}/{total}")
        print(f"{'=' * 65}\n")

        return self.failed == 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="E2E test: PrivacyLens ToolEmu sandbox integration"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use mock sandbox (no vLLM needed)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path/HF name of the model for vLLM auto-start")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for the vLLM payload")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for vLLM server")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID for vLLM")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--startup_timeout", type=int, default=300)
    parser.add_argument("--use_existing_server", action="store_true",
                        help="Don't start a new vLLM server")
    parser.add_argument("--no_real", action="store_true",
                        help="Skip real sandbox test even if vLLM is available")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed output")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    if args.model_name:
        os.environ["GUARD_MODEL"] = args.model_name

    # Start vLLM if needed
    server_proc = None
    server_log = None
    if not args.use_existing_server and not args.mock and args.model_path:
        print(f"Starting vLLM: {args.model_path}  gpu={args.gpu_id}  port={args.port}")
        server_proc, server_log = _start_vllm_server(
            args.model_path, args.port, args.gpu_id,
            args.gpu_mem_util, args.max_model_len,
        )

    try:
        if server_proc is not None:
            _wait_for_server(base_url, args.startup_timeout, server_proc, server_log)
        os.environ["GUARD_OPENAI_URL"] = base_url

        tester = PrivacyLensSandboxTester(
            mock=args.mock,
            verbose=args.verbose,
        )
        success = tester.run_all(skip_real=args.no_real or args.mock)
    finally:
        if server_proc is not None:
            os.killpg(server_proc.pid, signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except Exception:
                os.killpg(server_proc.pid, signal.SIGKILL)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
