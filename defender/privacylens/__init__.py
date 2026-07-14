"""PrivacyLens ToolEmu sandbox integration for SelfEvolvingPrivacyRL.

Provides the real PrivacyLens three-model (Agent + Simulator + Critiquer)
trajectory simulation, adapted to work with our local vLLM endpoint.

Dependencies (not installed by default — install when using PrivacyLens mode):
- ``langchain``  (required by PrivacyLens ToolEmu and our VLLM adapter)
- ``procoder``   (required by PrivacyLens prompt framework)
"""

# All imports are lazy — the module can be imported without langchain/PrivacyLens.
# Errors are raised only when you actually try to use the classes.


def _lazy_import(name: str):
    """Lazy attribute access with helpful error messages."""
    if name == "VLLMChatModel":
        try:
            from defender.privacylens.vllm_adapter import VLLMChatModel
        except ImportError as e:
            raise ImportError(
                "VLLMChatModel requires the 'langchain' package. "
                "Install it with: pip install langchain"
            ) from e
        return VLLMChatModel

    if name in ("PLSandboxConfig", "PrivacyLensSandbox", "run_privacylens_from_attacker"):
        try:
            from defender.privacylens.sandbox import (
                PLSandboxConfig,
                PrivacyLensSandbox,
                run_privacylens_from_attacker,
            )
        except ImportError as e:
            raise ImportError(
                "PrivacyLensSandbox requires langchain and the PrivacyLens "
                "codebase at /home/fangzibang/PrivacyLens. Ensure both are "
                "accessible."
            ) from e
        mapping = {
            "PLSandboxConfig": PLSandboxConfig,
            "PrivacyLensSandbox": PrivacyLensSandbox,
            "run_privacylens_from_attacker": run_privacylens_from_attacker,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name):
    return _lazy_import(name)


__all__ = [
    "VLLMChatModel",
    "PLSandboxConfig",
    "PrivacyLensSandbox",
    "run_privacylens_from_attacker",
]
